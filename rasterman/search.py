import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from enum import Enum
from itertools import permutations
from typing import NamedTuple
import cv2
import random
from scipy.spatial.transform import Rotation as sr

class Orientation(Enum):
    VERTICAL = 0
    HORIZONTAL = 1

class Stage(NamedTuple):
    grid: np.ndarray
    remaining_blocks: list[int]
    placed_blocks: list[tuple[int, tuple[int, int], Orientation]]

def occupied_spaces(length: int, placement: tuple):
    position, rotation = placement
    occupied = []
    match rotation:
        case Orientation.VERTICAL:
            occupied = [(position[1] - l, position[0]) for l in range(length)]
        case Orientation.HORIZONTAL:
            occupied = [(position[1], position[0] - l) for l in range(length)]
    return occupied

def place_into_new_grid(grid: np.ndarray, length: int, placement: tuple) -> np.ndarray:
    new_grid = grid.copy()
    for row, col in occupied_spaces(length, placement):
        new_grid[row, col] = 1
    return new_grid

def valid_placements(grid: np.ndarray, goal_img: np.ndarray, length: int):
    free = (goal_img == 1) & (grid != 1)
    H, W = free.shape
    configs = []
    if length <= H:
        ok = sliding_window_view(free, length, axis=0).all(axis=-1)  # (H-length+1, W)
        for top, col in zip(*(idx.tolist() for idx in np.nonzero(ok))):
            configs.append(((col, top + length - 1), Orientation.VERTICAL))
    if 1 < length <= W:
        ok = sliding_window_view(free, length, axis=1).all(axis=-1)  # (H, W-length+1)
        for row, left in zip(*(idx.tolist() for idx in np.nonzero(ok))):
            configs.append(((left + length - 1, row), Orientation.HORIZONTAL))
    return configs

def search(order_list: list, goal_img: np.ndarray):
    size = goal_img.shape[0]
    queue = [Stage(grid=np.zeros((size, size), dtype=np.uint8), remaining_blocks=order_list, placed_blocks=[])]
    visited = {}

    while len(queue) > 0:
        stage = queue.pop(0)

        if len(stage.remaining_blocks) == 0:
            return stage.grid, stage.placed_blocks
        else:
            length = stage.remaining_blocks[0]
            possible_stages = valid_placements(stage.grid, goal_img, length)
            if len(possible_stages) == 0:
                queue.append(
                    Stage(
                        grid=stage.grid,
                        remaining_blocks = stage.remaining_blocks[1:],
                        placed_blocks = stage.placed_blocks
                    )
                )
            else:
                for placement in possible_stages:
                    new_grid = place_into_new_grid(stage.grid, length, placement)
                    if new_grid.tobytes() not in visited:
                        new_stage = Stage(
                            grid=new_grid,
                            remaining_blocks=stage.remaining_blocks[1:],
                            placed_blocks = stage.placed_blocks + [(length, placement[0], placement[1])]
                        )
                        queue.append(new_stage)
                        visited[new_grid.tobytes()] = None

def render(goal_img: np.ndarray, solution: np.ndarray) -> str:
    def cells(row):
        return "".join("██" if cell else "  " for cell in row)

    lw, rw = 2 * goal_img.shape[1], 2 * solution.shape[1]
    top = "+" + "original".center(lw, "-") + "+" + "recreation".center(rw, "-") + "+"
    bottom = "+" + "-" * lw + "+" + "-" * rw + "+"
    body = [f"|{cells(g)}|{cells(s)}|" for g, s in zip(goal_img, solution)]
    return "\n".join([top, *body, bottom])


def poses(placements):
    centroids = []
    quats = []
    lens = []

    for placement in placements:
        length, position, rotation = placement
        match rotation:
            case Orientation.VERTICAL:
                offset = np.array([0.5, (-length / 2) + 1])
                rot = 0
            case Orientation.HORIZONTAL:
                offset = np.array([(-length / 2) + 1, 0.5])
                rot = 90
        
        centroid = np.array([position[0], position[1]]) + offset
        centroids.append(centroid.flatten())
        quat = sr.from_euler('xyz', (0, 0, rot), degrees=True).as_quat()
        quats.append(quat)
        lens.append(length)
    return centroids, quats, lens

def main(ones: int, twos: int, threes: int):
    # read target image
    goal_img: np.ndarray = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) # ty:ignore[invalid-assignment]
    goal_img = (goal_img < 128).astype(np.uint8)
    
    # create placement order
    nums = []
    for _ in range(threes): nums.append(3)
    for _ in range(twos): nums.append(2)
    for _ in range(ones): nums.append(1)
    
    # begin search
    res_img, res_placement = search(nums, goal_img)
    print(render(goal_img, res_img))
    print(poses(res_placement))

if __name__ == '__main__':
    main(6, 2, 4)