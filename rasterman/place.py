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

class Darwin:
    def __init__(self, parallel_generations: int, base_grid: Grid, base_img: np.ndarray):
        self.generations = [Generation(base_grid, base_img) for _ in range(parallel_generations)]

    def mutate(self):
        for gen in self.generations:
            gen.mutate()
    
    def iterate(self):
        gen_scores = [gen.score() for gen in self.generations]
        best_idx = gen_scores.index(max(gen_scores))
        self.generations = [deepcopy(self.generations[best_idx]) for _ in self.generations]


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

    # create starting grid
    base_grid = Grid(base_img.shape[0], 2, 1, 2, 0)

    # create genetic harness
    trainer = Darwin(4, base_grid, base_img)

    # run mutations
    cv2.namedWindow('centroids', cv2.WINDOW_NORMAL)

    for i in range(100):
        for _ in range(100):
            trainer.mutate()
            frames = [centroids_img(gen.grid) for gen in trainer.generations]
            gap = np.full((frames[0].shape[0], 10, 3), 128, dtype=np.uint8)
            separated = frames[0:1] + [f for frame in frames[1:] for f in (gap, frame)]
            cv2.imshow('centroids', np.hstack(separated))
            cv2.waitKey(1)
        trainer.iterate()
        print(f"gen {i}")

if __name__ == '__main__':
    main()