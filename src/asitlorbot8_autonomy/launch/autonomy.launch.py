import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
  pkg_tb3_sim = get_package_share_directory('asitlorbot8_autonomy')
  pkg_tb3_autonomy = get_package_share_directory('asitlorbot8_autonomy')

  autonomy_node_cmd = Node(
      package="asitlorbot8_autonomy",
      executable="autonomy_node",
      name="autonomy_node",
      parameters=[{
          "location_file": "/home/asimkumar/asitlorbot8_ws/src/asitlorbot8_autonomy/config/sim_house_locations.yaml"
      }]
  )

  ld = LaunchDescription()

  # Add the commands to the launch description
  ld.add_action(autonomy_node_cmd)

  return ld