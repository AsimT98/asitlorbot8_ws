#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import math

class PathPublisherNode(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        self.publisher_path = self.create_publisher(Path, 'path', 10)
        self.timer = self.create_timer(1.0, self.publish_path)
        self.create_path()

    def create_path(self):
        self.path = Path()
        self.path.header.frame_id = 'map'

        # Define the rectangle path (10m x 15m)
        waypoints = [
            (0.0, 0.0, 0.0),
            (0.004, 0.0, 0.0),
            (0.004, 0.004, math.pi/2),
            (0.0, 0.004, math.pi),
            (0.0, 0.0, -math.pi/2)
        ]

        for wp in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.orientation.z = math.sin(wp[2] / 2.0)
            pose.pose.orientation.w = math.cos(wp[2] / 2.0)
            self.path.poses.append(pose)

    def publish_path(self):
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.publisher_path.publish(self.path)

def main(args=None):
    rclpy.init(args=args)
    path_publisher_node = PathPublisherNode()
    rclpy.spin(path_publisher_node)
    path_publisher_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
