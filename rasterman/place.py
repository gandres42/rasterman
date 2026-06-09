import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from .grid import Block, Orientation, SillySearch
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray
from copy import deepcopy

# def centroids_img(grid: Grid):
#     centroids, quats, lens = grid.poses()
#     img = 1 - grid.image()
#     scale = 100
#     image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#     image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
#     for centroid, quat in zip(centroids, quats):
#         image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)
#     return image

def invert(img: np.ndarray) -> np.ndarray:
    return (img == 0).astype(float)

def block_factory(ones, twos, threes):
    blocks = []
    for i in range(ones):
        blocks.append(Block(1))
    for i in range(twos):
        blocks.append(Block(2))
    for i in range(threes):
        blocks.append(Block(3))
    return blocks

def main():
    # read target image
    goal_img: np.ndarray = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) # ty:ignore[invalid-assignment]
    for r in range(goal_img.shape[0]):
        for c in range(goal_img.shape[1]):
            if goal_img[r, c] >= 128:
                goal_img[r, c] = 0
            else:
                goal_img[r, c] = 1

    blocks: list[Block] = block_factory(6, 4, 3)
    print("searching...")
    grid = SillySearch.search(goal_img, blocks)
    print('yippee :)')

    print(goal_img)
    print(grid)

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 1:
                grid[r, c] = 255

    cv2.imshow('res', invert(grid))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()