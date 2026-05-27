import cv2
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation as sr
import rclpy
from rclpy.node import Node

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
                occupied = [(self.position[1] - l, self.position[0]) for l in range(self.length)]
            case Orientation.DOWN:
                occupied = [(self.position[1] + l, self.position[0]) for l in range(self.length)]
            case Orientation.LEFT:
                occupied = [(self.position[1], self.position[0] - l) for l in range(self.length)]
            case Orientation.RIGHT:
                occupied = [(self.position[1], self.position[0] + l) for l in range(self.length)]
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

    def centroids(self):
        centroids = []
        for block in self.blocks:
            match block.rotation:
                case Orientation.UP:
                    offset = np.array([0.5, (-block.length / 2) + 1])
                case Orientation.DOWN:
                    offset = np.array([0.5, block.length / 2])
                case Orientation.LEFT:
                    offset = np.array([(-block.length / 2) + 1, 0.5])
                case Orientation.RIGHT:
                    offset = np.array([block.length / 2, 0.5])
            
            centroid = np.array([block.position[0], block.position[1]]) + offset
            centroids.append(centroid)
        return centroids

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        # self.create_publisher(clear)

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# def main():
#     template = cv2.imread('result.jpg')
#     if template is None: return

#     grid = Grid(8)
#     test_block = Block(3)
#     test_block.set_pose((3,3), Orientation.DOWN)
#     test_block1 = Block(5)
#     test_block1.set_pose((2, 1), Orientation.RIGHT)
#     grid.add_block(test_block)
#     grid.add_block(test_block1)
#     img = grid.image()


#     scale = 100
#     image = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB)
#     image = cv2.resize(image, (img.shape[0] * scale, img.shape[1] * scale), interpolation=cv2.INTER_NEAREST)
#     for centroid in grid.centroids():
#         centroid = centroid.flatten()
#         image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=0, color=(0, 0, 255), thickness=25)
#     print(grid.centroids())

#     cv2.imshow("template", image)
#     cv2.waitKey(0)

# if __name__ == '__main__':
#     main()