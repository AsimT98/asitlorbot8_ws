import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    robot_description = ParameterValue( 
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("asitlorbot8_description"),
                    "urdf",
                    "asitlorbot8.urdf.xacro",
                ),
                " is_sim:=False"
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('asitlorbot8_description'),'launch','rsp.launch.py'
                )]), launch_arguments={'use_sim_time':'true'}.items()
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description,
             "use_sim_time": False},
            os.path.join(
                get_package_share_directory("asitlorbot8_controller"),
                "config",
                "asitlorbot8_controllers.yaml",
            ),
        ],
    )
    imu_node = Node(
            package='dextrobot_controller',
            executable='imu_sense.py',
            name='imu_sense',
            output='screen',
            respawn=True
        )
    return LaunchDescription(
        [
            robot_state_publisher_node,
            controller_manager,
            # imu_node
        ]
    )   