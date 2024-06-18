import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import serial

class ArduinoSerialNode(Node):
    def __init__(self):
        super().__init__('arduino_serial_node')
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        self.left_motor_sub = self.create_subscription(Int32, 'left_motor_speed', self.left_motor_callback, 10)
        self.right_motor_sub = self.create_subscription(Int32, 'right_motor_speed', self.right_motor_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.left_speed = 0
        self.right_speed = 0

    def left_motor_callback(self, msg):
        self.left_speed = msg.data
        self.serial_port.write(f'SETLEFT:{self.left_speed}\n'.encode())

    def right_motor_callback(self, msg):
        self.right_speed = msg.data
        self.serial_port.write(f'SETRIGHT:{self.right_speed}\n'.encode())

    def timer_callback(self):
        if self.serial_port.in_waiting:
            line = self.serial_port.readline().decode().strip()
            if line.startswith("LEFT:"):
                left_rpm = float(line.split(":")[1])
                self.get_logger().info(f'Left motor RPM: {left_rpm}')
            elif line.startswith("RIGHT:"):
                right_rpm = float(line.split(":")[1])
                self.get_logger().info(f'Right motor RPM: {right_rpm}')

def main(args=None):
    rclpy.init(args=args)
    arduino_serial_node = ArduinoSerialNode()
    rclpy.spin(arduino_serial_node)
    arduino_serial_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
