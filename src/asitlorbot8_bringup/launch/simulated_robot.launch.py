import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    gazebo = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("asitlorbot8_description"),
            "launch",
            "gazebo.launch.py"
        ),
    )

    rplidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_composition',
        name='rplidar_composition',
        output='screen',
        parameters=[{
            'serial_port': '/dev/ttyUSB0',  # Change if necessary
            'serial_baudrate': 115200,  # A1M8 typically uses 115200 baudrate
            'frame_id': 'laser_frame',
            'inverted': False,
            'angle_compensate': True,
        }]
    )
    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("asitlorbot8_controller"),
            "launch",
            "controller.launch.py"
        ),
        launch_arguments={
            "use_simple_controller": "False",
            "use_python": "False"
        }.items(),
    )
    
    joystick = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("asitlorbot8_controller"),
            "launch",
            "joystick_teleop.launch.py"
        ),
    )
    
    return LaunchDescription([
        gazebo,
        controller,
        rplidar_node
        # joystick,
    ])