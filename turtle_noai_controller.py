import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math

class Turtlesim_Speeder(Node):
    def __init__(self, nn):
        super().__init__(nn)

        self.pose_receiver = self.create_subscription(Pose, f'/{nn}/pose', self.callback_pose, 10)
        self.cmd_sender = self.create_publisher(Twist, f'/{nn}/cmd_vel', 10)

        self.target_route = [(2, 2), (9, 2), (9, 9), (2, 9)]
        self.current_index = 0
        # self.velocity = {x:self.get_best(x) for x in self.route}
        
        self.create_timer(0.1, self.main_loop)
        self.get_logger().info('🧠 AI海龟启动！')
        self.get_logger().info(f'目标路线: {self.target_route}')

    def callback_pose(self, msg):
        self.current_pose = msg

    def main_loop(self, diff_max=0.2):
        target = self.target_route[self.current_index]
        distance_diff = math.sqrt((self.current_pose.x - target[0])**2 + (self.current_pose.y - target[1])**2)
        if distance_diff < diff_max:
            self.get_logger().info(f"Reach {target} in point index {self.current_index}")
            self.current_index = (self.current_index + 1) % len(self.target_route)

        # v, w = self.velocity[target]
        v, w = self.get_velocity(target)
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_sender.publish(msg)

    def get_velocity(self, target_xy):
        """简单的神经网络替代 - 基于规则的智能控制"""
        target_x, target_y = target_xy
        
        # 计算目标向量
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 计算目标角度
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_pose.theta
        
        # 角度归一化（重要！）
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 智能控制策略
        if abs(angle_diff) > 0.3:  # 角度差太大，先转向
            v = 0.5  # 慢慢前进
            w = 2.0 if angle_diff > 0 else -2.0
        else:  # 角度基本对准，可以前进
            v = min(2.0, distance)  # 距离越远速度越快，但不超过最大值
            w = angle_diff * 1.5    # 微调角度
        
        return v, w

def main():
    rclpy.init()
    turtle_node = Turtlesim_Speeder('turtle1')
    try:
        rclpy.spin(turtle_node)
    except KeyboardInterrupt:
        print("Stop Speeding")

    turtle_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
