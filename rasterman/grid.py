import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from enum import Enum
from scipy.spatial.transform import Rotation as sr
from typing import Optional
import random
from copy import deepcopy

class Orientation(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Block:
    # Caches shared across all blocks; keyed so identical (length, grid) states reuse work.
    # reset_caches() must run between searches to avoid stale results across goals.
    _config_cache: dict = {}   # (length, grid_bytes) -> list[config]
    _goal_windows: dict = {}   # length -> (vertical_goal_mask, horizontal_goal_mask)

    def __init__(self, length, position = (0, 0), rotation = Orientation.UP):
        self.length = length
        self.position = position
        self.rotation = rotation

    @classmethod
    def reset_caches(cls):
        cls._config_cache = {}
        cls._goal_windows = {}

    def set_pose(self, new_position, new_rotation):
        self.position = new_position
        self.rotation = new_rotation

    def occupied_spaces(self):
        occupied = []
        match self.rotation:
            case Orientation.UP:
                occupied = [(self.position[1] - l, self.position[0]) for l in range(self.length)]
            case Orientation.DOWN:
                occupied = [(self.position[1] + l, self.position[0]) for l in range(self.length)]
            case Orientation.LEFT:
                occupied = [(self.position[1], self.position[0] - l) for l in range(self.length)]
            case Orientation.RIGHT:
                occupied = [(self.position[1], self.position[0] + l) for l in range(self.length)]
        return occupied

    def emplace(self, grid: np.ndarray):
        for space in self.occupied_spaces():
            grid[space[0], space[1]] = 1

    def valid_configurations(self, grid: np.ndarray, goal_img: np.ndarray):
        L = self.length
        key = (L, grid.tobytes())
        cache = Block._config_cache
        if key not in cache:
            H, W = grid.shape
            if L > 1:
                # The goal footprint per length is constant during a search; cache it once.
                if L not in Block._goal_windows:
                    gv = sliding_window_view(goal_img, (L, 1)).reshape(H - L + 1, W, L).sum(axis=2)
                    gh = sliding_window_view(goal_img, (1, L)).reshape(H, W - L + 1, L).sum(axis=2)
                    Block._goal_windows[L] = (gv == L, gh == L)
                goal_v, goal_h = Block._goal_windows[L]
                free_v = sliding_window_view(grid, (L, 1)).reshape(H - L + 1, W, L).sum(axis=2) == 0
                free_h = sliding_window_view(grid, (1, L)).reshape(H, W - L + 1, L).sum(axis=2) == 0
                vi = np.argwhere(free_v & goal_v)
                hi = np.argwhere(free_h & goal_h)
                cache[key] = (
                    [((int(j), int(i) + L - 1), Orientation.UP)   for i, j in vi] +
                    [((int(j), int(i)),          Orientation.DOWN) for i, j in vi] +
                    [((int(j) + L - 1, int(i)),  Orientation.LEFT) for i, j in hi] +
                    [((int(j), int(i)),          Orientation.RIGHT) for i, j in hi]
                )
            else:
                valid_mask = (grid == 0) & (goal_img == 1)
                cache[key] = [((int(c), int(r)), Orientation.UP) for r, c in np.argwhere(valid_mask)]
        return cache[key]

    def simple_str(self):
        return str(self.length)

class SillySearch:
    search_cache = {}
    goal_img = np.zeros((0, 0), dtype=np.uint8)

    @classmethod
    def search(cls, goal_img: np.ndarray, blocks: list[Block]):
        cls.goal_img = (np.asarray(goal_img) > 0).astype(np.uint8)
        cls.search_cache = {}
        Block.reset_caches()
        grid = np.zeros(cls.goal_img.shape, dtype=np.uint8)
        # Placements only land on empty goal cells, so a length-L block always lowers the
        # mismatch by exactly L. Track the score incrementally from this baseline.
        base_score = int(np.count_nonzero(cls.goal_img))
        best_grid, _ = cls._search(grid, blocks, base_score)
        return best_grid

    @classmethod
    def score(cls, grid):
        return int(np.count_nonzero(cls.goal_img != grid))

    @classmethod
    def _search(cls, parent_grid: np.ndarray, blocks: list[Block], parent_score: int):
        if len(blocks) == 0 or parent_score == 0:
            return parent_grid, parent_score

        key = (parent_grid.tobytes(), tuple(sorted(block.length for block in blocks)))
        cached = cls.search_cache.get(key)
        if cached is not None:
            return cached

        # No placement lands off-goal, so each block lowers the score by exactly its
        # length. The best any subtree can reach is therefore this bound (place them all);
        # once we hit it, further search cannot improve, so stop.
        lower_bound = parent_score - sum(block.length for block in blocks)

        # Fallback: leave the remaining blocks unplaced if none of them can be placed.
        best_grid = parent_grid
        best_score = parent_score
        tried_lengths = set()
        for block in blocks:
            # Same-length blocks are interchangeable; one representative covers them all.
            if block.length in tried_lengths:
                continue
            tried_lengths.add(block.length)
            new_blocklist = [subblock for subblock in blocks if subblock is not block]
            for config in block.valid_configurations(parent_grid, cls.goal_img):
                new_grid = np.copy(parent_grid)
                block.position = config[0]
                block.rotation = config[1]
                block.emplace(new_grid)
                child_grid, child_score = cls._search(new_grid, new_blocklist, parent_score - block.length)
                if child_score < best_score:
                    best_score = child_score
                    best_grid = child_grid
                    if best_score <= lower_bound:
                        break
            if best_score <= lower_bound:
                break

        cls.search_cache[key] = (best_grid, best_score)
        return best_grid, best_score

