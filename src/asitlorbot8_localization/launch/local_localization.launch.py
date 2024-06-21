# from launch import LaunchDescription
# from ament_index_python.packages import get_package_share_directory
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# import os

# def generate_launch_description():
    
#     static_transform_publisher = Node(
#         package="tf2_ros",
#         executable="static_transform_publisher",
#         arguments=["--x", "0", "--y", "0","--z", "0.14",
#                    "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
#                    "--frame-id", "base_footprint_ekf",
#                    "--child-frame-id", "imu_link_ekf"],
#     )
#     robot_localization = Node(
#         package="robot_localization",
#         executable="ekf_node",
#         name="ekf_filter_node",
#         output="screen",
#         parameters=[os.path.join(get_package_share_directory("asitlorbot8_localization"), "config", "ekf.yaml")],
#         # remappings=[("odometry/filtered", "odoms")]
#     )
#     imu_republisher_py = Node(
#         package="asitlorbot8_localization",
#         executable="imu_republisher.py"
#     )
#     rmse = Node(
#         package="asitlorbot8_localization",
#         executable="rmse.py"
#     )
#     return LaunchDescription([
#         static_transform_publisher,
#         robot_localization,
#         imu_republisher_py,
#         rmse
#     ])


from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import TimerAction
import os

def generate_launch_description():
    
    static_transform_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", "0", "--y", "0", "--z", "0.14",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "base_footprint_ekf",
                   "--child-frame-id", "imu_link_ekf"],
    )
    static_transform_publisher_tuned = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", "0", "--y", "0", "--z", "0.14",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "base_footprint_ekf_tuned",
                   "--child-frame-id", "imu_link_ekf_tuned"],
    )

    robot_localization = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[os.path.join(get_package_share_directory("asitlorbot8_localization"), "config", "ekf.yaml")],
        remappings=[("odometry/filtered", "odoms_one")]
    )
    robot_localization_tuned = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node_tuned",
        output="screen",
        parameters=[os.path.join(get_package_share_directory("asitlorbot8_localization"), "config", "ekf_tuned.yaml")],
        remappings=[("odometry/filtered", "odoms_tuned")]
    )
    imu_republisher_py = Node(
        package="asitlorbot8_localization",
        executable="imu_republisher.py"
    )
    imu_republisher_py_tuned = Node(
        package="asitlorbot8_localization",
        executable="imu_republisher_tuned.py"
    )
    rmse = Node(
        package="asitlorbot8_localization",
        executable="rmse.py"
    )
    path_publisher_node = Node(
        package='asitlorbot8_localization',
        executable='path_publisher_node.py',
        name='path_publisher_node',
        output='screen'
    )

    path_follower_node = Node(
        package='asitlorbot8_localization',
        executable='path_follower_node.py',
        name='path_follower_node',
        output='screen'
    )
    my_custom_node_1 = Node(
        package="asitlorbot8_localization",
        executable="circular_motion_1.py"
    )
    
    return LaunchDescription([
        static_transform_publisher,
        # static_transform_publisher_tuned,
        robot_localization,
        # robot_localization_tuned,
        imu_republisher_py,
        # imu_republisher_py_tuned,
        # TimerAction(
        #     period=1.0,
        #     actions=[
        #         rmse,
        #     ]
        # ),
        rmse,
        my_custom_node_1
        # path_publisher_node,
        # path_follower_node
    ])
