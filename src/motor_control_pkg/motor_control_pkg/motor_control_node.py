import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control_node')
        self.left_motor_pub = self.create_publisher(Int32, 'left_motor_speed', 10)
        self.right_motor_pub = self.create_publisher(Int32, 'right_motor_speed', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.setpoint = 100  # Example setpoint, adjust as needed

    def timer_callback(self):
        left_msg = Int32()
        right_msg = Int32()
        left_msg.data = self.setpoint
        right_msg.data = self.setpoint
        self.left_motor_pub.publish(left_msg)
        self.right_motor_pub.publish(right_msg)

def main(args=None):
    rclpy.init(args=args)
    motor_control_node = MotorControlNode()
    rclpy.spin(motor_control_node)
    motor_control_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
