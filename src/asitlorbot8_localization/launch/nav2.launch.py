#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joep Tool

import os
from time import sleep
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
  
  MAP_NAME='my_map_save_7_may'
  pkg_nav2_dir = get_package_share_directory('nav2_bringup')
  pkg_tb3_sim = get_package_share_directory('asitlorbot8_localization')

  use_sim_time = LaunchConfiguration('use_sim_time', default='True')
  autostart = LaunchConfiguration('autostart', default='True')
  
  default_map_path = PathJoinSubstitution(
        [FindPackageShare('asitlorbot8_localization'), 'maps', f'{MAP_NAME}.yaml']
    )
  
  nav2_sim_config_path = PathJoinSubstitution(
        [FindPackageShare('asitlorbot8_localization'), 'config', 'navigation_sim.yaml']
  )
  
  nav2_launch_cmd = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
          os.path.join(pkg_nav2_dir, 'launch', 'bringup_launch.py')
      ),
      launch_arguments={
          'map': default_map_path,
          'use_sim_time': use_sim_time,
          'autostart': autostart,
          'params_file': nav2_sim_config_path,
          
      }.items()
  )

  rviz_launch_cmd = Node(
    package="rviz2",
    executable="rviz2",
    name="rviz2",
    arguments=[
        '-d' + os.path.join(
            get_package_share_directory('nav2_bringup'),
            'rviz',
            'nav2_default_view.rviz'
        ),
        '--',
        '-r',  # Separate argument for ros args
        '__node:=/rviz2',
        '__params:=use_sim_time:=true'  # Set use_sim_time to true
    ]
  )
 
  static_transform_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["--x", "0", "--y", "0","--z", "0.15",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "base_footprint_ekf",
                   "--child-frame-id", "imu_link_ekf"],
    )
  
  set_init_amcl_pose_cmd = Node(
      package="asitlorbot8_localization",
      executable="set_init_amcl_pose.py",
      parameters=[{
          "x": 0.0,
          "y": 0.0,
      }]
  )

  ld = LaunchDescription()

  # Add the commands to the launch description
  ld.add_action(nav2_launch_cmd)
  ld.add_action(rviz_launch_cmd)
  ld.add_action(set_init_amcl_pose_cmd)
  ld.add_action(static_transform_publisher)

  return ld
