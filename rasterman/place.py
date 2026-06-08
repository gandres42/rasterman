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
    def __init__(self, base_grid: Grid):
        self.grid = deepcopy(base_grid)

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


def main():
    grid = Grid(7, 2)
    grid.add_block(GridBlock(length=2, position=(0, 0), rotation=Orientation.RIGHT))

    cv2.namedWindow('centroids', cv2.WINDOW_NORMAL)
    generation = Generation(grid)
    for i in range(0, 10):
        generation.mutate()

        centroids, quats, lens = generation.grid.poses()
        img = generation.grid.image()
        scale = 100
        image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        for centroid, quat in zip(centroids, quats):
            image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)

        # display image
        cv2.imshow('centroids', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()