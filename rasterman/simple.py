import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter
from .search import autoplace
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from cc_interfaces.msg import Block, StructurePlan  # ty:ignore[unresolved-import]
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray

CONSTRUCTION_SIZE = 1

class Rasterman(Node):
    def __init__(self):
        super().__init__('rasterman')
        self.bridge = CvBridge()

        self.struct_pub = self.create_publisher(StructurePlan, "structure_plan", 10)
        self.posearray_pub = self.create_publisher(PoseArray, "rasterman/poses", 10)
        self.viz_pub = self.create_publisher(Image, "rasterman/viz", 10)
        self.create_timer(0.1, self.pub)

        # Required parameter: declaring with only a type (no default) means the
        self.declare_parameter('image_path', Parameter.Type.STRING)
        image_path = self.get_parameter('image_path').get_parameter_value().string_value

        self.get_logger().info(f'starting initial plan')
        self.size, self.poses = autoplace(image_path, 6, 2, 2, stdout=False)
        self.get_logger().info(f'initial plan complete')
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