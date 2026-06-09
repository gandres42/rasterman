import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from enum import Enum
from itertools import permutations
from typing import NamedTuple
import cv2
import random

class Orientation(Enum):
    UP = 0
    LEFT = 1

class Stage(NamedTuple):
    grid: np.ndarray
    remaining_blocks: list[int]

def occupied_spaces(length: int, placement: tuple):
    position, rotation = placement
    occupied = []
    match rotation:
        case Orientation.UP:
            occupied = [(position[1] - l, position[0]) for l in range(length)]
        case Orientation.LEFT:
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
            configs.append(((col, top + length - 1), Orientation.UP))
    if 1 < length <= W:
        ok = sliding_window_view(free, length, axis=1).all(axis=-1)  # (H, W-length+1)
        for row, left in zip(*(idx.tolist() for idx in np.nonzero(ok))):
            configs.append(((left + length - 1, row), Orientation.LEFT))
    return configs

def search(order_list: list, goal_img: np.ndarray):
    size = goal_img.shape[0]
    queue = [Stage(grid=np.zeros((size, size), dtype=np.uint8), remaining_blocks=order_list)]
    visited = {}
    unique_solutions = {}

    while len(queue) > 0:
        stage = queue.pop(0)
        print(len(queue), stage.remaining_blocks)




        if len(stage.remaining_blocks) == 0:
            byte_key = stage.grid.tobytes()
            print(byte_key)
            if byte_key not in unique_solutions:
                unique_solutions[byte_key] = stage.grid
        else:
            length = stage.remaining_blocks[0]
            possible_stages = valid_placements(stage.grid, goal_img, length)
            if len(possible_stages) == 0:
                new_stage = Stage(
                    grid=stage.grid,
                    remaining_blocks=stage.remaining_blocks[1:]
                )
                queue.append(new_stage)
            else:
                
                for placement in possible_stages:
                    new_grid = place_into_new_grid(stage.grid, length, placement)
                    if new_grid.tobytes() not in visited:
                        new_stage = Stage(
                            grid=new_grid,
                            remaining_blocks=stage.remaining_blocks[1:]
                        )
                        queue.append(new_stage)
                        visited[new_grid.tobytes()] = None
        
    return random.choice(list(unique_solutions.values()))


def main(ones: int, twos: int, threes: int):
    # read target image
    goal_img: np.ndarray = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) # ty:ignore[invalid-assignment]
    for r in range(goal_img.shape[0]):
        for c in range(goal_img.shape[1]):
            if goal_img[r, c] >= 128:
                goal_img[r, c] = 0
            else:
                goal_img[r, c] = 1
    
    # make a list
    nums = []
    for _ in range(threes): nums.append(3)
    for _ in range(twos): nums.append(2)
    for _ in range(ones): nums.append(1)
    
    # begin loop
    print(search(nums, goal_img))

if __name__ == '__main__':
    main(60, 20, 100)