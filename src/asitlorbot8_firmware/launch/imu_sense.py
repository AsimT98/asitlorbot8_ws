#!/usr/bin/env python3

'''
        Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
'''
import smbus  # import SMBus module of I2C
from time import sleep  # import
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf_transformations import quaternion_from_euler
import time

# some MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47

# useful constants
M_PI = 3.14159265358979323846


class IMUPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')
        self.publisher_ = self.create_publisher(Imu, 'imu', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.yaw = 0.0
        self.bus = smbus.SMBus(1)  # or bus = smbus.SMBus(0) for older version boards
        self.device_address = 0x68  # MPU6050 device address
        self.MPU_Init()

    def MPU_Init(self):
        # write to sample rate register
        self.bus.write_byte_data(self.device_address, SMPLRT_DIV, 7)

        # Write to power management register
        self.bus.write_byte_data(self.device_address, PWR_MGMT_1, 1)

        # Write to Configuration register
        self.bus.write_byte_data(self.device_address, CONFIG, 0)

        # Write to Gyro configuration register
        self.bus.write_byte_data(self.device_address, GYRO_CONFIG, 24)

        # Write to interrupt enable register
        self.bus.write_byte_data(self.device_address, INT_ENABLE, 1)

    def read_raw_data(self, addr):
        # Accelero and Gyro value are 16-bit
        high = self.bus.read_byte_data(self.device_address, addr)
        low = self.bus.read_byte_data(self.device_address, addr + 1)

        # concatenate higher and lower value
        value = ((high << 8) | low)

        # to get signed value from mpu6050
        if value > 32768:
            value = value - 65536
        return value

    def publish_imu(self, roll, pitch, yaw, Ax, Ay, Az):
        # compose the message
        sensor_imu = Imu()
        sensor_imu.header.frame_id = "imu/data"
        sensor_imu.header.stamp = self.get_clock().now().to_msg()

        # convert from degrees to radians
        roll = roll * M_PI / 180
        pitch = pitch * M_PI / 180
        yaw = yaw * M_PI / 180

        q = quaternion_from_euler(roll, pitch, yaw)

        sensor_imu.orientation.x = q[0]
        sensor_imu.orientation.y = q[1]
        sensor_imu.orientation.z = q[2]
        sensor_imu.orientation.w = q[3]
        sensor_imu.linear_acceleration.x = float(Ax)
        sensor_imu.linear_acceleration.y = float(Ay)
        sensor_imu.linear_acceleration.z = float(Az)

        # publish the composed message
        self.publisher_.publish(sensor_imu)

    def timer_callback(self):
        # Read Accelerometer raw value
        acc_x = self.read_raw_data(ACCEL_XOUT_H)
        acc_y = self.read_raw_data(ACCEL_YOUT_H)
        acc_z = self.read_raw_data(ACCEL_ZOUT_H)

        # Read Gyroscope raw value
        gyro_x = self.read_raw_data(GYRO_XOUT_H)
        gyro_y = self.read_raw_data(GYRO_YOUT_H)
        gyro_z = self.read_raw_data(GYRO_ZOUT_H)

        # Full scale range +/- 250 degree/C as per sensitivity scale factor
        Ax = acc_x / 16384.0
        Ay = acc_y / 16384.0
        Az = acc_z / 16384.0

        Gx = gyro_x / 131.0
        Gy = gyro_y / 131.0
        Gz = gyro_z / 131.0

        # Calculate Pitch & Roll
        pitch = -(math.atan2(Ax, math.sqrt(Ay * Ay + Az * Az)) * 180.0) / M_PI
        roll = (math.atan2(Ay, Az) * 180.0) / M_PI

        # Ignore the gyro if our angular velocity does not meet our threshold
        if Gz > 1 or Gz < -1:
            Gz /= 20
            self.yaw += Gz

        # Keep our angle between 0-359 degrees
        if self.yaw < 0:
            self.yaw += 360
        elif self.yaw > 359:
            self.yaw -= 360

        self.publish_imu(roll, pitch, self.yaw, Ax, Ay, Az)


def main(args=None):
    rclpy.init(args=args)
    imu_publisher = IMUPublisher()
    rclpy.spin(imu_publisher)

    # Destroy the node explicitly
    imu_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
