import cv2
import numpy as np
from enum import Enum

cv2.namedWindow("template", cv2.WINDOW_NORMAL)

class Orientation(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Block:
    def __init__(self, length):
        self.length = length
        self.position = (0, 0)
        self.rotation = Orientation.UP

    def set_pose(self, new_position, new_rotation):
        self.position = new_position
        self.rotation = new_rotation

    def occupied_spaces(self):
        occupied = []
        match self.rotation:
            case Orientation.UP:
                occupied = [(self.position[0] - l, self.position[1]) for l in range(self.length)]
            case Orientation.DOWN:
                occupied = [(self.position[0] + l, self.position[1]) for l in range(self.length)]
            case Orientation.LEFT:
                occupied = [(self.position[0], self.position[1] - l) for l in range(self.length)]
            case Orientation.RIGHT:
                occupied = [(self.position[0], self.position[1] + l) for l in range(self.length)]
        return occupied

class Grid:
    def __init__(self, size):
        self.blocks: list[Block] = []
        self.size = size

    def add_block(self, block: Block):
        self.blocks.append(block)

    def image(self):
        img = np.ones((self.size, self.size))
        for block in self.blocks:
            for space in block.occupied_spaces():
                if space[0] >= 0 and space[0] < self.size and space[1] >= 0 and space[1] < self.size:
                    img[space[0], space[1]] = 0
        return img


def main():
    template = cv2.imread('result.jpg')
    if template is None: return

    grid = Grid(8)
    test_block = Block(3)
    test_block.set_pose((1, 1), Orientation.UP)
    grid.add_block(test_block)
    img = grid.image()

    cv2.imshow("template", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()