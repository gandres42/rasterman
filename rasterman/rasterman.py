import os
import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from .search import autoplace
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cc_interfaces.msg import Block, StructurePlan  # ty:ignore[unresolved-import]
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray

CONSTRUCTION_SIZE = 1
ONES = 1
TWOS = 1
THREES = 1

class Rasterman(Node):
    def __init__(self):
        super().__init__('rasterman')
        self.bridge = CvBridge()

        self.struct_pub = self.create_publisher(StructurePlan, "structure_plan", 10)
        self.posearray_pub = self.create_publisher(PoseArray, "rasterman_poses", 10)
        self.create_timer(0.1, self.pub)

        # Filename of an installed image; override with -p image:=<filename.jpg>
        self.declare_parameter('image', 'bill.jpg')
        image_name = self.get_parameter('image').get_parameter_value().string_value
        image_dir = os.path.join(get_package_share_directory('rasterman'), 'images')
        image_path = os.path.join(image_dir, image_name)


        self.get_logger().info(f'building plan...')
        self.size, self.poses = autoplace(image_path, ONES, TWOS, THREES, stdout=True)
        self.get_logger().info(f'plan complete')
        self.ratio = CONSTRUCTION_SIZE / self.size

    def pub(self):
        centroids, quats, lens = self.poses

        # create structure plan and posearray
        blocks = []
        poses = []
        for centroid, quat, length in zip(centroids, quats, lens):
            centroid_point = Point()
            centroid_point.x = (-centroid[1] + (self.size / 2)) * self.ratio
            centroid_point.y = (-centroid[0] + (self.size / 2)) * self.ratio

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

def main(args=None):
    rclpy.init(args=args)
    node = Rasterman()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()