import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from grid import GridBlock, Grid, Orientation
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray

grid = Grid(7, 2)
grid.add_block(GridBlock(length=1, position=(0, 0), rotation=Orientation.RIGHT))
grid.add_block(GridBlock(length=2, position=(1, 0), rotation=Orientation.LEFT))

centroids, quats, lens = grid.poses()

# create image
img = grid.image()
scale = 100
image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
for centroid, quat in zip(centroids, quats):
    image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)

# check validity
print(grid.valid_check())

# display image
cv2.namedWindow('centroids', cv2.WINDOW_NORMAL)
cv2.imshow('centroids', image)
cv2.waitKey(0)