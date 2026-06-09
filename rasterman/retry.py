import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from enum import Enum
from itertools import permutations
from typing import NamedTuple
import cv2

class Orientation(Enum):
    UP = 0
    LEFT = 1

class Block:
    def __init__(self, length, position = (0, 0), rotation = Orientation.UP):
        self.length = length
        self.position = position
        self.rotation = rotation

    def set_pose(self, new_position, new_rotation):
        self.position = new_position
        self.rotation = new_rotation

    def occupied_spaces(self):
        occupied = []
        match self.rotation:
            case Orientation.UP:
                occupied = [(self.position[1] - l, self.position[0]) for l in range(self.length)]
            case Orientation.LEFT:
                occupied = [(self.position[1], self.position[0] - l) for l in range(self.length)]
        return occupied

    def place_into_new_grid(self, grid: np.ndarray) -> np.ndarray:
        new_grid = grid.copy()
        for space in self.occupied_spaces():
            new_grid[space[0], space[1]] = 1
        return new_grid

    def set_pose(self, placement: tuple):
        self.position = placement[0]
        self.rotation = placement[1]

    def valid_placements(self, grid: np.ndarray, goal_img: np.ndarray):
        length = self.length
        # cells we are allowed to fill: goal pixels that are not occupied yet
        free = (goal_img == 1) & (grid != 1)
        H, W = free.shape

        configs = []

        # UP: vertical run of `length` cells, anchored at its bottom cell.
        # A window starting at row `top` (extending down) is valid when every
        # cell in it is free; the anchor is the bottom cell, so row = top + length - 1.
        if length <= H:
            ok = sliding_window_view(free, length, axis=0).all(axis=-1)  # (H-length+1, W)
            for top, col in zip(*(idx.tolist() for idx in np.nonzero(ok))):
                configs.append(((col, top + length - 1), Orientation.UP))

        # LEFT: horizontal run of `length` cells, anchored at its rightmost cell.
        # Length 1 is identical to the UP placement, so only add it for length > 1.
        if 1 < length <= W:
            ok = sliding_window_view(free, length, axis=1).all(axis=-1)  # (H, W-length+1)
            for row, left in zip(*(idx.tolist() for idx in np.nonzero(ok))):
                configs.append(((left + length - 1, row), Orientation.LEFT))

        return configs

class Stage(NamedTuple):
    grid: np.ndarray
    remaining_blocks: list[str]

def order_proc(order_list: list, goal_img: np.ndarray):
    size = goal_img.shape[0]
    queue = [Stage(grid=np.zeros((size, size)), remaining_blocks=order_list)]
    visited = {}
    next_block = Block(0)

    while len(queue) > 0:
        stage = queue.pop(0)

        # check if reached end
        if len(stage.remaining_blocks) == 0:
            continue

        # print(len(queue), stage.remaining_blocks)
        next_block.length = int(stage.remaining_blocks[0])
        valid_placements = next_block.valid_placements(stage.grid, goal_img)
        
        for placement in valid_placements:
            next_block.set_pose(placement)
            new_grid = next_block.place_into_new_grid(stage.grid)
            if new_grid.tobytes() not in visited:
                new_stage = Stage(
                    grid=new_grid,
                    remaining_blocks=stage.remaining_blocks[1:]
                )
                queue.append(new_stage)
                visited[new_grid.tobytes()] = None
        



def main(ones: int, twos: int, threes: int):
    # read target image
    goal_img: np.ndarray = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) # ty:ignore[invalid-assignment]
    for r in range(goal_img.shape[0]):
        for c in range(goal_img.shape[1]):
            if goal_img[r, c] >= 128:
                goal_img[r, c] = 0
            else:
                goal_img[r, c] = 1
    
    # get all unique block placement orders
    base_str = ""
    for _ in range(ones): base_str = base_str + "1" 
    for _ in range(twos): base_str = base_str + "2" 
    for _ in range(threes): base_str = base_str + "3" 

    orders = list(permutations(base_str))
    print(len(orders))
    order_proc(list(orders[0]), goal_img)

if __name__ == '__main__':
    main(6, 2, 2)