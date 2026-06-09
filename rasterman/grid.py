import cv2
import numpy as np
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
            case Orientation.DOWN:
                occupied = [(self.position[1] + l, self.position[0]) for l in range(self.length)]
            case Orientation.LEFT:
                occupied = [(self.position[1], self.position[0] - l) for l in range(self.length)]
            case Orientation.RIGHT:
                occupied = [(self.position[1], self.position[0] + l) for l in range(self.length)]
        return occupied

class Grid:
    def __init__(self, real_size, goal_img: np.ndarray):
        self.blocks: list[Block] = []
        self.size = goal_img.shape[0]
        self.ratio = real_size / self.size
        self.config_cache: dict[int, list] = {}
        self.goal_img = goal_img

    def image(self):
        img = np.ones((self.size, self.size))
        for block in self.blocks:
            for space in block.occupied_spaces():
                if space[0] >= 0 and space[0] < self.size and space[1] >= 0 and space[1] < self.size:
                    img[space[0], space[1]] = 0
        return (img == 0).astype(float)

    def poses(self):
        centroids = []
        quats = []
        lens = []

        offset = np.array([0, 0, 0])
        rot = np.eye(3)
        for block in self.blocks:
            match block.rotation:
                case Orientation.UP:
                    offset = np.array([0.5, (-block.length / 2) + 1])
                    rot = 0
                case Orientation.RIGHT:
                    offset = np.array([block.length / 2, 0.5])
                    rot = 270
                case Orientation.DOWN:
                    offset = np.array([0.5, block.length / 2])
                    rot = 180
                case Orientation.LEFT:
                    offset = np.array([(-block.length / 2) + 1, 0.5])
                    
                    rot = 90
            
            centroid = np.array([block.position[0], block.position[1]]) + offset
            centroids.append(centroid.flatten())
            quat = sr.from_euler('xyz', (0, 0, rot), degrees=True).as_quat()
            quats.append(quat)
            lens.append(block.length)
        return centroids, quats, lens

    def valid_check(self):
        count = np.zeros((self.size, self.size))
        for block in self.blocks:
            for space in block.occupied_spaces():
                if space[0] >= 0 and space[0] < self.size and space[1] >= 0 and space[1] < self.size:
                    if count[space[0], space[1]] == 1:
                        return False
                    else:
                        count[space[0], space[1]] = 1
                else:
                    return False
        return True

    def add_block(self, block: Block):
        self.blocks.append(block)

    def valid_configurations(self, block_length: int):
        configs = []
        occupied = set()
        for block in self.blocks:
            for space in block.occupied_spaces():
                occupied.add(space)

        for col in range(self.size):
            for row in range(self.size):
                for rotation in Orientation:
                    spaces = Block(block_length, (col, row), rotation).occupied_spaces()
                    if all(0 <= r < self.size and 0 <= c < self.size and (r, c) not in occupied and self.goal_img[r, c] == 1 for r, c in spaces):
                        configs.append(((col, row), rotation))

        return configs

    def score(self):
        return np.sum(np.abs(self.goal_img - self.image()))


# class RecursiveSearch:
#     def __init__(self, one_count: int, two_count: int, three_count: int):
#         if one_count == 0 and two_count == 0 and three_count = 0

#         self.children = []
