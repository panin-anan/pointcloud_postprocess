from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointcloud_postprocess',
            namespace='Grinding_calculation',
            executable='thickness_calc',
            name='postprocess_calc1'
        )
    ])