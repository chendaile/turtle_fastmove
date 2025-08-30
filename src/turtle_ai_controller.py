import time
from math import sqrt, atan2, pi
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np
import json
import argparse

class optimized_para():
    def __init__(self, route_name, contine_bool=False):
        self.best_param_path = 'output/best_params_' + route_name + '.json'
        if not contine_bool:
            with open('config/init_arg.json', 'r') as f:
                config = json.load(f)
            self.params = config["params"]
        else:
            with open(self.best_param_path, 'r') as f:
                config = json.load(f)
            self.params = config["best_params"]

        with open('config/init_arg.json', 'r') as f:
            config = json.load(f)
        self.params_range = config["params_range"]
        self.current_paramList = np.array(list(self.params.values()))

    def save_best_params(self, lap_time):
        save_data = {
            'best_time': lap_time,
            'best_params': self.params.copy()
        }
        try:
            with open(self.best_param_path, 'r') as f:
                old_data = json.load(f)
            if lap_time >= old_data['best_time']:
                return False 
        except:
            pass
        
        with open(self.best_param_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        return True 
    
    def update_params(self, param_updates):
        old_params = self.params.copy()
        
        if isinstance(param_updates, dict):    
            for param_name, update in param_updates.items():
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(update, min_val, max_val)
        elif isinstance(param_updates, np.ndarray):
            for i, param_name in enumerate(self.params.keys()):
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(param_updates[i], min_val, max_val)

        self.current_paramList = np.array(list(self.params.values()))
        
        return old_params 

    def generate_candidate_params(self):
        current = self.current_paramList
        noise = np.random.normal(0, 0.1, size=current.shape)
        candidate = current + noise
        
        param_names = list(self.params.keys())
        for i, name in enumerate(param_names):
            min_val, max_val = self.params_range[name]
            candidate[i] = np.clip(candidate[i], min_val, max_val)
                
        return candidate

class network():
    def __init__(self):
        # 改进网络结构：9 -> 16 -> 12 -> 8 -> 1 (增加层数和宽度)
        self.input_size = 9
        self.hidden1_size = 16
        self.hidden2_size = 12  
        self.hidden3_size = 8
        self.output_size = 1

        # 使用Xavier初始化，比随机初始化更好
        self.w1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(2.0/self.input_size)
        self.b1 = np.zeros(self.hidden1_size)
        
        self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0/self.hidden1_size)
        self.b2 = np.zeros(self.hidden2_size)
        
        self.w3 = np.random.randn(self.hidden2_size, self.hidden3_size) * np.sqrt(2.0/self.hidden2_size)
        self.b3 = np.zeros(self.hidden3_size)
        
        self.w4 = np.random.randn(self.hidden3_size, self.output_size) * np.sqrt(2.0/self.hidden3_size)
        self.b4 = np.zeros(self.output_size)

        self.learning_rate = 0.01

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, input_data: np.ndarray):
        self.input_data = input_data
        
        self.z1 = input_data @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = self.relu(self.z3)
        
        self.z4 = self.a3 @ self.w4 + self.b4
        self.output = self.z4  # 直接线性输出

        return self.output
    
    def backward(self, target_output: np.ndarray):
        # 输出层误差
        output_error = self.output - target_output
        delta4 = output_error  # 线性输出，导数为1
        
        error3 = delta4 @ self.w4.T
        delta3 = error3 * self.relu_derivative(self.z3)
        
        error2 = delta3 @ self.w3.T
        delta2 = error2 * self.relu_derivative(self.z2)
        
        error1 = delta2 @ self.w2.T
        delta1 = error1 * self.relu_derivative(self.z1)
        
        self.w4 -= self.learning_rate * np.outer(self.a3, delta4)
        self.b4 -= self.learning_rate * delta4
        
        self.w3 -= self.learning_rate * np.outer(self.a2, delta3)
        self.b3 -= self.learning_rate * delta3
        
        self.w2 -= self.learning_rate * np.outer(self.a1, delta2)
        self.b2 -= self.learning_rate * delta2
        
        self.w1 -= self.learning_rate * np.outer(self.input_data, delta1)
        self.b1 -= self.learning_rate * delta1

        loss = float(output_error[0] ** 2)
        return loss

    def train_network(self, data, logger=None):
        total_loss = 0
        sample_count = 0
        
        for params, lap_time in data[-5:]:
            predicted = self.forward(params)
            loss = self.backward(np.array([lap_time]))
            total_loss += loss
            sample_count += 1
            
            if logger:
                logger.info(f"预测:{predicted[0]:.1f}s, 实际:{lap_time:.1f}s, 误差:{abs(predicted[0]-lap_time):.1f}s")
    
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

        self.distance_thres = 0.3
        self.total_route = 0
        self.total_node = -1
        self.start_time = time.time()
        self.training_data = []
        self.havePrint = False
        self.haveStart = False

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

        if hasattr(self, 'lap_start_time') and time.time() - self.lap_start_time > 30:
            self.duty_index = (self.duty_index + 1) % len(self.duty)

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
