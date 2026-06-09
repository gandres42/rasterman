import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from .grid import Block, Grid, Orientation
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray
from copy import deepcopy

def centroids_img(grid: Grid):
    centroids, quats, lens = grid.poses()
    img = 1 - grid.image()
    scale = 100
    image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    for centroid, quat in zip(centroids, quats):
        image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)
    return image

class SillySearch:
    cache = {}

    @classmethod
    def recursive_search(cls, grid: Grid, one_count: int, two_count: int, three_count: int) -> Grid:
        if one_count == 0 and two_count == 0 and three_count == 0:
            return grid
        else:
            children = []
            if one_count > 0:
                for config in grid.valid_configurations(1):
                    new_grid = deepcopy(grid)
                    new_grid.add_block(Block(1, config[0], config[1]))
                    children.append(cls.recursive_search(new_grid, one_count - 1, two_count, three_count))
            if two_count > 0:
                for config in grid.valid_configurations(2):
                    new_grid = deepcopy(grid)
                    new_grid.add_block(Block(2, config[0], config[1]))
                    children.append(cls.recursive_search(new_grid, one_count, two_count - 1, three_count))
            if three_count > 0:
                for config in grid.valid_configurations(3):
                    new_grid = deepcopy(grid)
                    new_grid.add_block(Block(3, config[0], config[1]))
                    children.append(cls.recursive_search(new_grid, one_count, two_count, three_count - 1))
            
            if children == []:
                return grid
            else:
                min_score = np.inf
                favorite_child = children[0]
                for child in children:
                    score = child.score()
                    if score < min_score:
                        favorite_child = child
                return favorite_child

def invert(img: np.ndarray) -> np.ndarray:
    return (img == 0).astype(float)

def main():
    # read target image
    goal_img = (cv2.imread('result.jpg', cv2.IMREAD_GRAYSCALE) == 0).astype(float)  # ty:ignore[unresolved-attribute]

    grid = Grid(2,goal_img)
    best = SillySearch.recursive_search(grid, 6, 2, 2)
    cv2.imshow('res', invert(best.image()))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()