import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class EncoderSubscriberNode(Node):
    def __init__(self):
        super().__init__('encoder_subscriber_node')
        self.left_motor_sub = self.create_subscription(Int32, 'left_motor_speed', self.left_motor_callback, 10)
        self.right_motor_sub = self.create_subscription(Int32, 'right_motor_speed', self.right_motor_callback, 10)

    def left_motor_callback(self, msg):
        left_speed = msg.data
        self.get_logger().info(f'Left motor speed: {left_speed}')

    def right_motor_callback(self, msg):
        right_speed = msg.data
        self.get_logger().info(f'Right motor speed: {right_speed}')

def main(args=None):
    rclpy.init(args=args)
    encoder_subscriber_node = EncoderSubscriberNode()
    rclpy.spin(encoder_subscriber_node)
    encoder_subscriber_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
