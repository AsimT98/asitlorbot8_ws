#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf_transformations import euler_from_quaternion
import math

class PathFollowerNode(Node):
    def __init__(self):
        super().__init__('path_follower_node')
        self.publisher_cmd_vel = self.create_publisher(Twist, 'diff_cont/cmd_vel_unstamped', 10)
        self.subscription_path = self.create_subscription(Path, 'path', self.path_callback, 10)
        self.path = None
        self.current_waypoint_index = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.reached_goal = False

    def path_callback(self, msg):
        self.path = msg
        self.current_waypoint_index = 0
        self.reached_goal = False

    def timer_callback(self):
        if self.path is None or self.reached_goal:
            return

        if self.current_waypoint_index >= len(self.path.poses):
            self.reached_goal = True
            self.stop_robot()
            return

        target_pose = self.path.poses[self.current_waypoint_index]
        self.move_towards_waypoint(target_pose)

    def move_towards_waypoint(self, target_pose):
        target_x = target_pose.pose.position.x
        target_y = target_pose.pose.position.y
        current_position = self.get_current_position()
        if current_position is None:
            return

        distance = math.hypot(target_x - current_position[0], target_y - current_position[1])
        angle_to_target = math.atan2(target_y - current_position[1], target_x - current_position[0])
        current_yaw = current_position[2]

        angle_difference = angle_to_target - current_yaw
        angle_difference = math.atan2(math.sin(angle_difference), math.cos(angle_difference))

        twist_msg = Twist()

        if abs(angle_difference) > 0.1:
            twist_msg.angular.z = 0.35 * angle_difference
        elif distance > 0.1:
            twist_msg.linear.x = 0.2 * distance
        else:
            self.current_waypoint_index += 1
            return

        self.publisher_cmd_vel.publish(twist_msg)

    def get_current_position(self):
        # For simplicity, using dummy current position.
        # Replace this with actual position from odometry or localization system.
        return (0, 0, 0)  # (x, y, yaw)

    def stop_robot(self):
        twist_msg = Twist()
        self.publisher_cmd_vel.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    path_follower_node = PathFollowerNode()
    rclpy.spin(path_follower_node)
    path_follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
