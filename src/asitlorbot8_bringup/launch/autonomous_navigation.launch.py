import os
import subprocess
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription

def generate_launch_description():
    map_file = os.path.join(
        get_package_share_directory('mobile_robot_autonomous_navigation'),
        'maps', 'layout.yaml'
    )
    model_file = os.path.join(
        get_package_share_directory('mobile_robot_autonomous_navigation'),
        'urdf', 'mobile_robot.urdf.xacro'
    )
    controllers_config_file = os.path.join(
        get_package_share_directory('mobile_robot_autonomous_navigation'),
        'config', 'controllers.yaml'
    )
    joint_limits_config_file = os.path.join(
        get_package_share_directory('mobile_robot_autonomous_navigation'),
        'config', 'joint_limits.yaml'
    )
    rviz_config_file = os.path.join(
        get_package_share_directory('mobile_robot_autonomous_navigation'),
        'config', 'nav_config.rviz'
    )

    robot_description_cmd = [
        'xacro',
        'python',
        model_file
    ]

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file]
    )

    map_server_node = Node(
        package='map_server',
        executable='map_server',
        name='map_server',
        arguments=[map_file]
    )

    rplidar_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rplidar_ros'), 'launch', 'rplidar.launch.py')
        )
    )

    amcl_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mobile_robot_autonomous_navigation'), 'launch', 'amcl.launch.py')
        )
    )

    move_base_launch = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('mobile_robot_autonomous_navigation'), 'launch', 'move_base.launch.py')
        )
    )

    controller_spawner_node = Node(
        package='controller_manager',
        executable='spawner',
        name='controller_spawner',
        output='screen',
        arguments=[
            '/mobile_robot/joints_update',
            '/mobile_robot/mobile_base_controller'
        ]
    )

    return LaunchDescription([
        
        # robot_state_publisher_node,
        rviz_node,
        # controller_spawner_node,
        # map_server_node,
        rplidar_launch,
        amcl_launch,
        move_base_launch
    ])

