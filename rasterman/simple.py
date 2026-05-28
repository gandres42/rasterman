import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from .grid import GridBlock, Grid, Orientation
from cc_interfaces.msg import Block, StructurePlan
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray

CONSTRUCTION_SIZE = 2

class Rasterman(Node):
    def __init__(self):
        super().__init__('rasterman')
        self.bridge = CvBridge()

        self.struct_pub = self.create_publisher(StructurePlan, "rasterman/structure_plan", 10)
        self.posearray_pub = self.create_publisher(PoseArray, "rasterman/poses", 10)
        self.viz_pub = self.create_publisher(Image, "rasterman/viz", 10)
        self.create_timer(0.1, self.pub)

        self.grid = Grid(7, 1)
        self.grid.add_block(GridBlock(length=1, position=(3, 3), rotation=Orientation.UP))
        self.grid.add_block(GridBlock(length=3, position=(1, 1), rotation=Orientation.DOWN))
        self.grid.add_block(GridBlock(length=1, position=(6, 6), rotation=Orientation.RIGHT))

    def pub(self):
        centroids, quats, lens = self.grid.poses()

        # create structure plan and posearray
        blocks = []
        poses = []
        for centroid, quat, length in zip(centroids, quats, lens):
            centroid_point = Point()
            centroid_point.x = (-centroid[1] + (self.grid.size / 2)) * self.grid.ratio
            centroid_point.y = (-centroid[0] + (self.grid.size / 2)) * self.grid.ratio

            centroid_rot = Quaternion()
            centroid_rot.x = quat[0]
            centroid_rot.y = quat[1]
            centroid_rot.z = quat[2]
            centroid_rot.w = quat[3]

            block_pose: Pose = Pose()
            block_pose.position = centroid_point
            block_pose.orientation = centroid_rot
            poses.append(block_pose)

            block_posestamped = PoseStamped()
            block_posestamped.pose = block_pose

            block_msg = Block()
            block_msg.pose = block_posestamped
            block_msg.type = length - 1
            blocks.append(block_msg)
        
        blocklist = StructurePlan()
        blocklist.blocks = blocks
        self.struct_pub.publish(blocklist)

        pose_array = PoseArray()
        pose_array.header.frame_id = "rasterman"
        pose_array.poses = poses
        self.posearray_pub.publish(pose_array)

        # create image
        img = self.grid.image()
        scale = 100
        image = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        for centroid, quat in zip(centroids, quats):
            image = cv2.circle(image, (int(centroid[0] * scale), int(centroid[1] * scale)), radius=5, color=(0, 255, 0), thickness=-1)
        self.viz_pub.publish(self.bridge.cv2_to_imgmsg(image, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = Rasterman()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()