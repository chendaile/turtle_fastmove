import time
from math import sqrt, atan2, pi
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import numpy as np

class optimized_para():
    def __init__(self):
        self.params = {
            'angle_threshold': 0.1,          # 角度阈值
            'turn_speed': 3.0,               # 转向速度
            'speed_angle_factor': 2.0,       # 转向时速度衰减因子
            'min_turn_speed': 0.2,           # 转向时最小速度
            
            # 微调参数
            'fine_tune_gain': 2.0,           # 微调增益
            
            # 距离控制参数
            'speed_distance_factor': 1.5,    # 速度-距离系数
            'max_speed': 3.0,                # 最大速度
            
            # 减速参数
            'slow_distance': 0.5,            # 开始减速的距离
            'slow_factor': 0.3,              # 减速系数
        }

        self.params_range = {
            'angle_threshold': (0.02, 1),          
            'turn_speed': (1, 10),               
            'speed_angle_factor': (0.5, 5.0),       
            'min_turn_speed': (0.01, 1),           
            
            'fine_tune_gain': (0.05, 5.0),           
            
            'speed_distance_factor': (0.5, 5),    
            'max_speed': (1, 5.0),                
            
            'slow_distance': (0.1, 5),            
            'slow_factor': (0.1, 3),              
        }

        self.current_paramList = np.array(list(self.params.values()))

    def update_params(self, param_updates):
        if isinstance(param_updates, dict):    
            for param_name, update in param_updates.items():
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(update, min_val, max_val)
        elif isinstance(param_updates, np.ndarray):
            for i, param_name in enumerate(self.params.keys()):
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(param_updates[i], min_val, max_val)

    def generate_candidate_params(self):
        current = self.current_paramList
        noise = np.random.normal(0, 0.1, size=current.shape)
        candidate = current + noise
        
        param_names = list(self.param.params.keys())
        for i, name in enumerate(param_names):
            min_val, max_val = self.params_range[name]
            candidate[i] = np.clip(candidate[i], min_val, max_val)
                
        return candidate

class network():
    def __init__(self):
        self.layer01_size = (9, 8)
        self.layer02_size = (8, 1)

        self.weight01 = np.random.rand(*self.layer01_size)
        self.bias01 = np.random.rand(self.layer01_size[1])
        self.weight02 = np.random.rand(*self.layer02_size)
        self.bias02 = np.random.rand(self.layer02_size[1])

        self.input = input

        self.learning_rate = 0.0001

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, input: np.ndarray):
        self.z1 = input @ self.weight01 + self.bias01
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.weight02 + self.bias02
        self.a2 = self.tanh(self.z2)

        return self.a2
    
    def backward(self, target_output: np.ndarray):
        output_error = self.a2 - target_output
        delta2 = output_error * self.tanh_derivative(self.z2)
        
        error1 = delta2 @ self.w2.T
        delta1 = error1 * self.relu_derivative(self.z1)
        
        self.weight02 -= self.learning_rate * np.outer(self.a2, delta2)
        self.bias02 -= self.learning_rate * delta2
        
        self.weight01 -= self.learning_rate * np.outer(self.a1, delta1)
        self.bias01 -= self.learning_rate * delta1

    def train_network(self, data):
        for params, lap_time in data[-5:]:  # 使用最近5次数据
            self.forward(params)
            self.backward(np.array(list(lap_time)))
    
class turtle_node(Node):
    def __init__(self):
        super().__init__('turtle1')
        self.param = optimized_para()
        self.brain = network()
        self.get_logger().info("Start turtle node!")

        self.receiver = self.create_subscription(
            Pose, '/turtle1/pose', self.callback_pose, 10
        )
        self.cmd_sender = self.create_publisher(
            Twist, '/turtle1/cmd_vel', 10
        )

        self.duty = {1:(2, 2), 
                     2:(9, 2), 
                     3:(9, 9), 
                     4:(2, 9)}
        self.duty_index = 0

        self.distance_thres = 0.3
        self.total_route = 0
        self.total_node = 0
        self.time_tmp = 0
        self.lap_start_time = time.time()
        self.training_data = []

        self.create_timer(0.1, self.mainloop)
        self.get_logger().info("Start turtle main loop")

    def callback_pose(self, pose):
        if hasattr(self, 'pos'):
            add_route = sqrt((pose.x - self.pos.x)**2 + (pose.y - self.pos.y)**2)
            self.total_route += add_route

        self.des_point_pos = self.duty[self.duty_index]
        gap_x, gap_y = pose.x - self.des_point_pos[0], pose.y - self.des_point_pos[1]
        self.distance = sqrt(gap_x**2 + gap_y**2)

        self.gap_theta = atan2(gap_y, gap_x)
        angle_diff = self.gap_theta - pose.theta
        while angle_diff > pi:
            angle_diff -= 2 * pi
        while angle_diff < -pi:
            angle_diff += 2 * pi
        self.angle_diff = angle_diff

        if self.distance < self.distance_thres:
            self.get_logger().info(f"Successfully reach {self.des_point_pos}")
            self.duty_index = (self.duty_index + 1) % len(self.duty)
            self.total_node += 1

        self.pos = pose
        self.total_time = time.time() - self.start_time
        
    def send_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_sender.publish(msg)
    
    def print_status(self):
        self.get_logger().info(f"Total route: {self.total_route}")
        self.get_logger().info(f"Total time: {self.total_time}")
        self.get_logger().info(f"Average velocity: {self.total_route / self.total_time}")
        self.get_logger().info(f"Have run {self.total_node} nodes")
        self.get_logger().info(f"Now at ({self.pos.x:.1f}, {self.pos.y:.1f})")
        self.get_logger().info(f"Target at {self.des_point_pos}")

    def get_best_cmd(self, distance, angle_diff):
        #Get best v and w ... using self.param.params
        
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

    def mainloop(self):
        if self.total_node > 0 and self.total_node % 4 == 0:
            lap_time = time.time() - self.lap_start_time
            self.training_data.append((self.param.current_paramList.copy(), lap_time))
            
            if len(self.training_data) > 5:
                self.brain.train_network(self.training_data)
                self.optimize_parameters() 
            
            self.lap_start_time = time.time()
            self.print_status()

        v, m = self.get_best_cmd(self.distance, self.angle_diff)
        self.send_cmd(v, m)

    def optimize_parameters(self):
        best_params = None
        best_predicted_time = float('inf')
        
        for _ in range(10):  # 尝试10个随机参数组合
            # 在当前参数附近生成候选参数
            candidate_params = self.param.generate_candidate_params()
            predicted_time = self.brain.forward(candidate_params)[0]
            
            if predicted_time < best_predicted_time:
                best_predicted_time = predicted_time
                best_params = candidate_params
        
        if best_params is not None:
            self.param.update_params(best_params)
    
def main():
    rclpy.init()
    turtlesim = turtle_node()

    try:
        rclpy.spin(turtlesim)
    except KeyboardInterrupt:
        print("Interrupted by keyboard")

    turtlesim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
