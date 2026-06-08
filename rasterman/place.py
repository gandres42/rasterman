import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from .grid import GridBlock, Grid, Orientation
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray
from copy import deepcopy
import random

class Generation:
    def __init__(self, base_grid: Grid, base_img: np.ndarray):
        self.grid = deepcopy(base_grid)
        self.base_img = base_img

    def mutate(self):
        lucky_fella = random.choice(self.grid.blocks)
        new_config = random.choice(
            self.grid.valid_configurations(
                lucky_fella.length,
                lucky_fella.position,
                lucky_fella.rotation
            )
        )
        lucky_fella.set_pose(new_config[0], new_config[1])

    # def score
    def score(self):
        img = self.grid.image()
        return np.sum(np.abs(self.base_img - img))

def centroids_img(grid: Grid):
    centroids, quats, lens = grid.poses()
    img = 1 - grid.image()
    scale = 100
    image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    for centroid, quat in zip(centroids, quats):
        image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)
    return image

def main():
    # read target image
    base_img = cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE)
    base_img = (base_img == 0).astype(float)  # ty:ignore[unresolved-attribute]

    # update inventory
    grid = Grid(base_img.shape[0], 2, 1, 2, 0)
    # grid.add_block(GridBlock(length=2, position=(0, 0), rotation=Orientation.RIGHT))
    # grid.add_block(GridBlock(length=1, position=(1, 0), rotation=Orientation.RIGHT))

    cv2.namedWindow('centroids', cv2.WINDOW_NORMAL)
    generation = Generation(grid, base_img)
    for i in range(0, 100):
        generation.mutate()
        print(generation.score())
        cv2.imshow('centroids', centroids_img(generation.grid))
        cv2.waitKey(10)

if __name__ == '__main__':
    main()