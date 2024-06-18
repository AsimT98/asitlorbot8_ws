from setuptools import setup, find_packages

package_name = 'motor_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asimkumar',
    maintainer_email='asimtailor1998@gmail.com',
    description='ROS2 package for motor control with Arduino Nano',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_control_node = motor_control_pkg.motor_control_node:main',
            'encoder_subscriber_node = motor_control_pkg.encoder_subscriber_node:main',
            'arduino_serial_node = motor_control_pkg.arduino_serial_node:main',
        ],
    },
)
