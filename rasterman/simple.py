import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from .grid import GridBlock, Grid, Orientation
from cc_interfaces.msg import Block, StructurePlan
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.bridge = CvBridge()

        self.struct_pub = self.create_publisher(StructurePlan, "rasterman/structure_plan", 10)
        self.viz_pub = self.create_publisher(Image, "rasterman/viz", 10)
        self.create_timer(0.1, self.pub)

        self.grid = Grid(7)
        self.grid.add_block(GridBlock(length=1, position=(3, 3), rotation=Orientation.RIGHT))

    def pub(self):
        centroids, quats, lens = self.grid.poses()

        # create structure plan
        blocks = []
        for centroid, quat, length in zip(centroids, quats, lens):
            centroid_point = Point()
            centroid_point.x = centroid[1]
            centroid_point.y = centroid[0]

            centroid_rot = Quaternion()
            centroid_rot.x = quat[0]
            centroid_rot.y = quat[1]
            centroid_rot.z = quat[2]
            centroid_rot.w = quat[3]

            block_pose: Pose = Pose()
            block_pose.position = centroid_point
            block_pose.orientation = centroid_rot

            block_posestamped = PoseStamped()
            block_posestamped.pose = block_pose

            block_msg = Block()
            block_msg.pose = block_posestamped
            block_msg.type = length - 1
            blocks.append(block_msg)
        blocklist = StructurePlan()
        blocklist.blocks = blocks
        self.struct_pub.publish(blocklist)

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
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()