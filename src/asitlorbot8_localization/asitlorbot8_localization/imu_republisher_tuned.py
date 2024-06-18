#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Imu

imu_pub = None

def imuCallback(imu):
    global imu_pub
    imu.header.frame_id = "base_footprint_ekf_tuned"
    imu_pub.publish(imu)

def main(args=None):
    global imu_pub
    rclpy.init(args=args)
    node = Node('imu_republisher_node_tuned')
    time.sleep(1)
    imu_pub = node.create_publisher(Imu, "imu_ekf_tuned", 10)
    imu_sub = node.create_subscription(Imu, "imu/data", imuCallback, 10) #imu/data
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()