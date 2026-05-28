import cv2
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as sr

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

    def add_block(self, block: GridBlock):
        self.blocks.append(block)

    def image(self):
        img = np.ones((self.size, self.size))
        for block in self.blocks:
            for space in block.occupied_spaces():
                if space[0] >= 0 and space[0] < self.size and space[1] >= 0 and space[1] < self.size:
                    img[space[0], space[1]] = 0
        return img

    def poses(self):
        centroids = []
        quats = []
        lens = []

        # center_offset = np.array([-self.size / 2, - self.size / 2])
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
