import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mpu6050',
            executable='mpu6050_node',
            name='mpu6050_node',
            output='screen',
            parameters=[{
                'i2c_address': 0x68,
                'frame_id': 'imu_link',
                'use_mag': False,
            }],
            remappings=[('/imu/data_raw', '/imu_data')],
        ),
    ])
