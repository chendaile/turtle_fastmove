import time
from math import sqrt, atan2, pi
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import json
import argparse
# from turtle_network import network
from turtle_para import optimized_para
from turtle_network_sci import network

class turtle_node(Node):
    def __init__(self, continue_bool, route_name):
        super().__init__('turtle1')
        self.param = optimized_para(route_name, continue_bool)
        self.brain = network()
        self.get_logger().info("Start turtle node!")

        self.receiver = self.create_subscription(
            Pose, '/turtle1/pose', self.callback_pose, 10
        )
        self.cmd_sender = self.create_publisher(
            Twist, '/turtle1/cmd_vel', 10
        )

        with open("config/routes.json", 'r') as f:
            routes = json.load(f)
        self.duty = routes["routes"][route_name]
        description = routes["route_descriptions"][route_name]
        self.get_logger().info(description)

        self.duty_index = 0
        self.len_route = len(self.duty)

        self.distance_thres = 0.2
        self.total_route = 0
        self.total_node = -1
        self.start_time = time.time()
        self.training_data = []
        self.havePrint = False
        self.haveStart = False
        self.haveFix = False

        self.create_timer(0.1, self.mainloop)
        self.get_logger().info("Start turtle main loop")

    def callback_pose(self, pose):
        if hasattr(self, 'pos'):
            add_route = sqrt((pose.x - self.pos.x)**2 + (pose.y - self.pos.y)**2)
            self.total_route += add_route

        self.des_point_pos = self.duty[str(self.duty_index)]
        gap_x, gap_y = self.des_point_pos[0] - pose.x, self.des_point_pos[1] - pose.y 
        self.distance = sqrt(gap_x**2 + gap_y**2)

        self.gap_theta = atan2(gap_y, gap_x)
        angle_diff = self.gap_theta - pose.theta
        while angle_diff > pi:
            angle_diff -= 2 * pi
        while angle_diff < -pi:
            angle_diff += 2 * pi
        self.angle_diff = angle_diff

        if self.distance < self.distance_thres:
            self.duty_index = (self.duty_index + 1) % len(self.duty)
            self.total_node += 1
            self.haveFix = False

        self.pos = pose
        self.total_time = time.time() - self.start_time
        
    def send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_sender.publish(msg)

    def get_best_cmd(self, distance, angle_diff):
        # 角度控制
        if abs(angle_diff) > self.param.params['angle_threshold']:
            # 需要转向
            w = self.param.params['turn_speed'] if angle_diff > 0 else -self.param.params['turn_speed']
            v = max(self.param.params['min_turn_speed'], 
                   self.param.params['max_speed'] - abs(angle_diff) * self.param.params['speed_angle_factor'])
        else:
            # 可以直走
            w = angle_diff * self.param.params['fine_tune_gain']
            v = min(self.param.params['max_speed'], distance * self.param.params['speed_distance_factor'])
        
        # 距离控制
        if distance < self.param.params['slow_distance']:
            v *= self.param.params['slow_factor']

        return v, w
    
    def optimize_parameters(self):
        best_params = None
        best_predicted_time = float('inf')
        
        for i in range(20): 
            candidate_params = self.param.generate_candidate_params()
            predicted_time = self.brain.forward(candidate_params)[0]
            
            if predicted_time < best_predicted_time:
                best_predicted_time = predicted_time
                best_params = candidate_params

        old_params = self.param.update_params(best_params)
        self.get_logger().info(f"优化完成! 预测改进时间: {best_predicted_time:.2f}秒")
        
        param_names = list(self.param.params.keys())
        self.get_logger().info("参数变化详情:")
        
        for name in param_names:
            old_val = old_params[name]
            new_val = self.param.params[name]
            change = new_val - old_val
            change_pct = (change / old_val * 100) if old_val != 0 else 0
            
            if abs(change) > 0.01: 
                self.get_logger().info(f"  {name}: {old_val:.3f} → {new_val:.3f} "
                                    f"(变化: {change:+.3f}, {change_pct:+.1f}%)")

    def mainloop(self):
        if self.total_node == 0 and not self.haveStart:
            self.haveStart = True
            self.lap_start_time = time.time()
            
        if self.total_node > 0 and self.total_node % self.len_route == 0 and not self.havePrint:
            self.havePrint = True
            lap_time = time.time() - self.lap_start_time
            self.lap_start_time = time.time()

            self.get_logger().info(f"完成一圈，用时: {lap_time:.2f}秒")
            self.training_data.append((self.param.current_paramList.copy(), lap_time))
            
            if len(self.training_data) > 5:
                self.get_logger().info("=== 网络训练结果 ===")
                self.brain.train_network(self.training_data, self.get_logger())
                self.get_logger().info("=== 参数优化 ===")
                self.optimize_parameters()
        if self.total_node % self.len_route != 0:
            self.havePrint = False

        if hasattr(self, 'distance'):
            v, m = self.get_best_cmd(self.distance, self.angle_diff)
            self.send_cmd(v, m)

        if hasattr(self, 'lap_start_time') and time.time() - self.lap_start_time > 30 and not self.haveFix:
            self.duty_index = (self.duty_index + 1) % len(self.duty)
            self.haveFix = True

        try:
            if self.param.save_best_params(lap_time):
                self.get_logger().info("🎉 新最佳成绩已保存!")
        except:
            pass

def main():
    rclpy.init()
    parser = argparse.ArgumentParser(description='Please select the required turtle running parameters')
    parser.add_argument("--contin", '-u', action='store_true', 
                        default=False, help='Whether to continue from a given params')
    parser.add_argument("--route", '-r', default="simple",
                        help="The turtle's route among simple, figure_eight, spiral," \
                        "star, complex_polygon, s_curve, maze, diamond, triangle, zigzag")

    args = parser.parse_args()
    continue_bool = args.contin
    route_name = args.route

    turtlesim = turtle_node(continue_bool, route_name)

    try:
        rclpy.spin(turtlesim)
    except KeyboardInterrupt:
        print("Interrupted by keyboard")

    turtlesim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
