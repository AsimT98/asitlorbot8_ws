import os
from os import pathsep
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml

def generate_launch_description():
    
    asitlorbot8_description = get_package_share_directory("asitlorbot8_description")
    asitlorbot8_description_prefix = get_package_prefix("asitlorbot8_description")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    model_arg = DeclareLaunchArgument(name="model", default_value=os.path.join(
                                        asitlorbot8_description, "urdf", "asitlorbot8.urdf.xacro"
                                        ),
                                      description="Absolute path to robot urdf file"
    )
    
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    model_path = os.path.join(asitlorbot8_description, "models")
    model_path += pathsep + os.path.join(asitlorbot8_description_prefix, "share")

    env_var = SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path)

    robot_description = ParameterValue(Command(["xacro ", LaunchConfiguration("model")]),value_type=str)
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}] #
    )

    start_robot_state_publisher_cmd = IncludeLaunchDescription(
        os.path.join(asitlorbot8_description, 'launch', 'rsp.launch.py'),
    )
    
    # start_gazebo_server = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")
    #     )
    # )
    gazebo_params_file = os.path.join(asitlorbot8_description, 'config', 'gazebo_params.yaml')
    with open(gazebo_params_file, 'r') as file:
        gazebo_params = yaml.safe_load(file)['gazebo']['ros__parameters']
    start_gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzserver.launch.py")
        ),
        launch_arguments={
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file
        }.items()
    )
    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gzclient.launch.py")
        )
    )

    spawn_robot = Node(package="gazebo_ros", executable="spawn_entity.py",
                        arguments=["-entity", "asitlorbot8",
                                   "-topic", "robot_description",
                                   "-x", x_pose,
                                    "-y", y_pose
                                  ],
                        output="screen",
    )

    return LaunchDescription([
        env_var,
        model_arg,
        start_gazebo_server,
        start_gazebo_client,
        robot_state_publisher_node,
        # start_robot_state_publisher_cmd,
        spawn_robot
    ])