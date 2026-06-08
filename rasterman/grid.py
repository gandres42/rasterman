import cv2
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as sr
from typing import Optional

class Orientation(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GridBlock:
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
    def __init__(self, size, real_size):
        self.blocks: list[GridBlock] = []
        self.size = size
        self.ratio = real_size / size
        self._config_cache: dict[int, list] = {}

    def add_block(self, block: GridBlock):
        self.blocks.append(block)

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

    def valid_configurations(self, block_length: int, current_pos: Optional[tuple], current_rot: Optional[Orientation]):
        # get all possible configs
        configs = []
        if block_length in self._config_cache:
            configs = self._config_cache[block_length]
        else:
            occupied = set()
            for block in self.blocks:
                for space in block.occupied_spaces():
                    occupied.add(space)

            for col in range(self.size):
                for row in range(self.size):
                    for rotation in Orientation:
                        candidate = GridBlock(block_length, (col, row), rotation)
                        spaces = candidate.occupied_spaces()
                        if all(
                            0 <= r < self.size and 0 <= c < self.size and (r, c) not in occupied
                            for r, c in spaces
                        ):
                            configs.append(((col, row), rotation))

            self._config_cache[block_length] = configs
        
        # optionally filter only mutations
        filtered_configs = []
        if current_pos is not None and current_rot is not None:
            for config in configs:
                # same position, changed rotation
                if config[0] == current_pos and config[1] != current_rot:
                    filtered_configs.append(config)
                # same rotation, different position
                elif config[0] != current_pos and config[1] == current_rot and np.sum(np.abs(np.array(config[0]) - np.array(current_pos))) == 1:
                    filtered_configs.append(config)

        return filtered_configs if current_pos is not None and current_rot is not None else configs