import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from simple_pid import PID

class PIDControllerNode(Node):
    def __init__(self):
        super().__init__('pid_controller')
        
        # Declare PID parameters
        self.declare_parameter('Kp', 1.0)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.0)

        # Get initial PID parameters
        Kp = self.get_parameter('Kp').value
        Ki = self.get_parameter('Ki').value
        Kd = self.get_parameter('Kd').value

        # Initialize PID controller
        self.pid = PID(Kp, Ki, Kd, setpoint=0)

        # Create subscribers and publishers
        self.subscription = self.create_subscription(
            Float64, 'input', self.input_callback, 10)
        self.publisher = self.create_publisher(Float64, 'output', 10)

        # Create a timer to periodically publish control output
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info('PID Controller Node has been started.')

    def input_callback(self, msg):
        self.pid.setpoint = msg.data

    def timer_callback(self):
        # Read the current process variable (e.g., motor speed)
        current_value = 0.0 # Replace with actual value
        control = self.pid(current_value)
        self.publisher.publish(Float64(data=control))

def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
