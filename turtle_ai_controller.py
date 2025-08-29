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

    # def update_params(self, param_updates):
    #     if isinstance(param_updates, dict):    
    #         for param_name, update in param_updates.items():
    #             min_val, max_val = self.params_range[param_name]
    #             self.params[param_name] = np.clip(update, min_val, max_val)
    #     elif isinstance(param_updates, np.ndarray):
    #         for i, param_name in enumerate(self.params.keys()):
    #             min_val, max_val = self.params_range[param_name]
    #             self.params[param_name] = np.clip(param_updates[i], min_val, max_val)

    #     self.current_paramList = np.array(list(self.params.values()))

    def update_params(self, param_updates):
        old_params = self.params.copy()  # 保存旧参数
        
        if isinstance(param_updates, dict):    
            for param_name, update in param_updates.items():
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(update, min_val, max_val)
        elif isinstance(param_updates, np.ndarray):
            for i, param_name in enumerate(self.params.keys()):
                min_val, max_val = self.params_range[param_name]
                self.params[param_name] = np.clip(param_updates[i], min_val, max_val)

        self.current_paramList = np.array(list(self.params.values()))
        
        return old_params  # 返回旧参数用于日志

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

        self.learning_rate = 0.01  # 提高学习率

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, input_data: np.ndarray):
        self.input_data = input_data
        
        # 层1
        self.z1 = input_data @ self.w1 + self.b1
        self.a1 = self.relu(self.z1)
        
        # 层2  
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = self.relu(self.z2)
        
        # 层3
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = self.relu(self.z3)
        
        # 输出层(不用激活函数，因为时间可以是任何正值)
        self.z4 = self.a3 @ self.w4 + self.b4
        self.output = self.z4  # 直接线性输出

        return self.output
    
    def backward(self, target_output: np.ndarray):
        # 输出层误差
        output_error = self.output - target_output
        delta4 = output_error  # 线性输出，导数为1
        
        # 第3层误差
        error3 = delta4 @ self.w4.T
        delta3 = error3 * self.relu_derivative(self.z3)
        
        # 第2层误差
        error2 = delta3 @ self.w3.T
        delta2 = error2 * self.relu_derivative(self.z2)
        
        # 第1层误差
        error1 = delta2 @ self.w2.T
        delta1 = error1 * self.relu_derivative(self.z1)
        
        # 更新所有权重和偏置
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
        
        avg_loss = total_loss / sample_count if sample_count > 0 else 0
        if logger:
            logger.info(f"平均训练损失: {avg_loss:.3f}")
        
        return avg_loss
    
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

        self.duty = {0:(2, 2), 
                     1:(9, 2), 
                     2:(9, 9), 
                     3:(2, 9)}
        self.duty_index = 0

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

        self.des_point_pos = self.duty[self.duty_index]
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

    # def optimize_parameters(self):
    #     best_params = None
    #     best_predicted_time = float('inf')
        
    #     for _ in range(10): 
    #         candidate_params = self.param.generate_candidate_params()
    #         predicted_time = self.brain.forward(candidate_params)[0]
            
    #         if predicted_time < best_predicted_time:
    #             best_predicted_time = predicted_time
    #             best_params = candidate_params
        
    #     if best_params is not None:
    #         self.param.update_params(best_params)

    def optimize_parameters(self):
        best_params = None
        best_predicted_time = float('inf')
        
        for i in range(10): 
            candidate_params = self.param.generate_candidate_params()
            predicted_time = self.brain.forward(candidate_params)[0]
            
            if predicted_time < best_predicted_time:
                best_predicted_time = predicted_time
                best_params = candidate_params
        
        if best_params is not None:
            # 更新参数并获取旧参数
            old_params = self.param.update_params(best_params)
            
            self.get_logger().info(f"优化完成! 预测改进时间: {best_predicted_time:.2f}秒")
            
            # 显示参数变化
            param_names = list(self.param.params.keys())
            self.get_logger().info("参数变化详情:")
            
            for name in param_names:
                old_val = old_params[name]
                new_val = self.param.params[name]
                change = new_val - old_val
                change_pct = (change / old_val * 100) if old_val != 0 else 0
                
                if abs(change) > 0.01:  # 只显示变化较大的参数
                    self.get_logger().info(f"  {name}: {old_val:.3f} → {new_val:.3f} "
                                        f"(变化: {change:+.3f}, {change_pct:+.1f}%)")
        else:
            self.get_logger().info("未找到更好的参数组合")

    def mainloop(self):
        if self.total_node == 0 and not self.haveStart:
            self.haveStart = True
            self.lap_start_time = time.time()
            
        if self.total_node > 0 and self.total_node % 4 == 0 and not self.havePrint:
            self.havePrint = True
            lap_time = time.time() - self.lap_start_time
            self.lap_start_time = time.time()

            self.get_logger().info(f"完成一圈，用时: {lap_time:.2f}秒")
            self.training_data.append((self.param.current_paramList.copy(), lap_time))
            
            if len(self.training_data) > 5:
                self.get_logger().info("=== 网络训练结果 ===")
                avg_loss = self.brain.train_network(self.training_data, self.get_logger())
                self.get_logger().info("=== 参数优化 ===")
                self.optimize_parameters()
        if self.total_node % 4 != 0:
            self.havePrint = False

        if hasattr(self, 'distance'):
            v, m = self.get_best_cmd(self.distance, self.angle_diff)
            self.send_cmd(v, m)
    
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
