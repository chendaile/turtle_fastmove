import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import numpy as np
import json
import os

class ParameterizedPolicy:
    """参数化的控制策略"""
    
    def __init__(self):
        # 可优化的策略参数（这些就是神经网络要学习的！）
        self.params = {
            # 角度控制参数
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
        
        # 参数边界（防止参数超出合理范围）
        self.param_bounds = {
            'angle_threshold': (0.02, 0.3),
            'turn_speed': (0.2, 5.0),
            'speed_angle_factor': (0.5, 10.0),
            'min_turn_speed': (0.1, 5.0),
            'fine_tune_gain': (0.5, 10.0),
            'speed_distance_factor': (0.1, 3.0),
            'max_speed': (1.0, 10.0),
            'slow_distance': (0.2, 5.0),
            'slow_factor': (0.1, 5),
        }

    def generate_action(self, distance, angle_diff):
        """使用参数化策略生成动作"""
        
        # 角度控制
        if abs(angle_diff) > self.params['angle_threshold']:
            # 需要转向
            w = self.params['turn_speed'] if angle_diff > 0 else -self.params['turn_speed']
            v = max(self.params['min_turn_speed'], 
                   self.params['max_speed'] - abs(angle_diff) * self.params['speed_angle_factor'])
        else:
            # 可以直走
            w = angle_diff * self.params['fine_tune_gain']
            v = min(self.params['max_speed'], distance * self.params['speed_distance_factor'])
        
        # 距离控制
        if distance < self.params['slow_distance']:
            v *= self.params['slow_factor']
        
        return v, w

    def update_params(self, param_updates):
        """更新参数并确保在边界内"""
        for param_name, update in param_updates.items():
            if param_name in self.params:
                new_value = self.params[param_name] + update
                # 限制在边界内
                min_val, max_val = self.param_bounds[param_name]
                self.params[param_name] = np.clip(new_value, min_val, max_val)

    def get_param_vector(self):
        """将参数转换为向量（用于神经网络）"""
        return np.array(list(self.params.values()))

class PolicyParameterNetwork:
    """优化策略参数的神经网络"""
    
    def __init__(self):
        # 网络结构：状态(5) -> 隐藏(12) -> 隐藏(8) -> 参数调整(9)
        self.input_size = 5   # dx, dy, distance, angle_diff, speed
        self.hidden1_size = 12
        self.hidden2_size = 8
        self.output_size = 9  # 9个策略参数的调整值
        
        # 初始化权重（小一些，因为我们只是在微调参数）
        self.w1 = np.random.randn(self.input_size, self.hidden1_size) * 0.1
        self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * 0.1
        self.w3 = np.random.randn(self.hidden2_size, self.output_size) * 0.05
        
        self.b1 = np.random.randn(self.hidden1_size) * 0.01
        self.b2 = np.random.randn(self.hidden2_size) * 0.01
        self.b3 = np.random.randn(self.output_size) * 0.01
        
        # 学习参数
        self.learning_rate = 0.0001  # 更小的学习率，因为我们在微调
        
        # 存储前向传播结果
        self.z1 = None
        self.a1 = None
        self.z2 = None  
        self.a2 = None
        self.z3 = None
        self.a3 = None
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, state):
        """前向传播：预测参数调整"""
        # 归一化输入
        normalized_state = np.array([
            state[0] / 10.0,  # dx
            state[1] / 10.0,  # dy  
            state[2] / 10.0,  # distance
            state[3] / math.pi,  # angle_diff
            state[4] / 5.0    # speed
        ])
        
        # 第一隐藏层
        self.z1 = np.dot(normalized_state, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 第二隐藏层
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # 输出层（参数调整值，范围[-1, 1]）
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.tanh(self.z3) * 0.1  # 限制调整幅度
        
        return self.a3
    
    def backward(self, state, target_adjustments):
        """反向传播"""
        # 输出层误差
        output_error = self.a3 - target_adjustments
        delta3 = output_error * self.tanh_derivative(self.z3) * 0.1
        
        # 第二隐藏层误差
        error2 = np.dot(delta3, self.w3.T)
        delta2 = error2 * self.relu_derivative(self.z2)
        
        # 第一隐藏层误差
        error1 = np.dot(delta2, self.w2.T)
        delta1 = error1 * self.relu_derivative(self.z1)
        
        # 归一化状态
        normalized_state = np.array([
            state[0] / 10.0, state[1] / 10.0, state[2] / 10.0,
            state[3] / math.pi, state[4] / 5.0
        ])
        
        # 更新权重
        self.w3 -= self.learning_rate * np.outer(self.a2, delta3)
        self.b3 -= self.learning_rate * delta3
        
        self.w2 -= self.learning_rate * np.outer(self.a1, delta2)
        self.b2 -= self.learning_rate * delta2
        
        self.w1 -= self.learning_rate * np.outer(normalized_state, delta1)
        self.b1 -= self.learning_rate * delta1

class ParameterOptimizationTurtle(Node):
    def __init__(self, turtle_name):
        super().__init__(f'{turtle_name}_param_opt')
        
        # ROS2设置
        self.pose_receiver = self.create_subscription(
            Pose, f'/{turtle_name}/pose', self.pose_callback, 10)
        self.cmd_sender = self.create_publisher(
            Twist, f'/{turtle_name}/cmd_vel', 10)
        
        # 控制参数
        self.target_route = [(2, 2), (9, 2), (9, 9), (2, 9)]
        self.current_index = 0
        self.current_pose = None
        
        # 参数化策略和神经网络
        self.policy = ParameterizedPolicy()
        self.param_network = PolicyParameterNetwork()
        
        # 性能评估
        self.last_distance = None
        self.performance_history = []
        self.targets_reached = 0
        
        # 训练数据收集
        self.training_states = []
        self.training_adjustments = []
        
        # 控制循环
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('🎛️ 参数化策略优化海龟启动！')
        self.get_logger().info(f'📊 初始参数: {self.policy.params}')

    def pose_callback(self, msg):
        self.current_pose = msg
    
    def evaluate_performance(self, old_distance, new_distance, reached_target):
        """评估当前性能"""
        performance = 0
        
        # 距离改善奖励
        if old_distance is not None:
            performance += (old_distance - new_distance) * 10
        
        # 到达目标大奖励
        if reached_target:
            performance += 100
        
        # 距离惩罚
        performance -= new_distance * 0.5
        
        return performance
    
    def generate_parameter_adjustments(self, performance):
        """基于性能生成参数调整建议"""
        # 简单的性能导向调整策略
        adjustments = {}
        
        if performance > 50:  # 表现很好
            # 可以尝试提高速度
            adjustments['max_speed'] = 0.1
            adjustments['speed_distance_factor'] = 0.05
        elif performance < -10:  # 表现不好
            # 更保守的策略
            adjustments['angle_threshold'] = -0.01  # 更早转向
            adjustments['turn_speed'] = -0.1  # 转向更慢
            adjustments['slow_factor'] = -0.02  # 更早减速
        
        # 转换为向量形式
        param_names = list(self.policy.params.keys())
        adjustment_vector = np.array([
            adjustments.get(name, 0.0) for name in param_names
        ])
        
        return adjustment_vector
    
    def control_loop(self):
        """主控制循环"""
        if not self.current_pose:
            return
        
        target = self.target_route[self.current_index]
        
        # 计算状态
        dx = target[0] - self.current_pose.x
        dy = target[1] - self.current_pose.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.current_pose.theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        current_speed = math.sqrt(self.current_pose.linear_velocity**2 + 
                                 self.current_pose.angular_velocity**2)
        
        current_state = [dx, dy, distance, angle_diff, current_speed]
        
        # 检查是否到达目标
        reached_target = distance < 0.3
        if reached_target:
            self.targets_reached += 1
            self.get_logger().info(f'🎯 到达目标 {target} (第{self.targets_reached}个)')
            self.current_index = (self.current_index + 1) % len(self.target_route)
            return
        
        # 评估性能并收集训练数据
        if self.last_distance is not None:
            performance = self.evaluate_performance(self.last_distance, distance, reached_target)
            self.performance_history.append(performance)
            
            # 生成参数调整建议
            target_adjustments = self.generate_parameter_adjustments(performance)
            
            # 收集训练数据
            self.training_states.append(current_state.copy())
            self.training_adjustments.append(target_adjustments.copy())
            
            # 限制训练数据大小
            if len(self.training_states) > 1000:
                self.training_states.pop(0)
                self.training_adjustments.pop(0)
        
        # 使用神经网络预测参数调整
        if len(self.training_states) > 50:
            param_adjustments = self.param_network.forward(current_state)
            # 将调整应用到策略参数
            param_names = list(self.policy.params.keys())
            adjustment_dict = {name: param_adjustments[i] for i, name in enumerate(param_names)}
            self.policy.update_params(adjustment_dict)
        
        # 使用当前策略生成控制命令
        v, w = self.policy.generate_action(dx, dy, distance, angle_diff, current_speed)
        
        # 发送控制命令
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(w)
        self.cmd_sender.publish(cmd)
        
        self.last_distance = distance
        
        # 定期训练网络
        if len(self.training_states) > 100 and len(self.training_states) % 50 == 0:
            self.train_parameter_network()
        
        # 定期显示状态
        if len(self.performance_history) > 0 and len(self.performance_history) % 100 == 0:
            avg_performance = np.mean(self.performance_history[-50:])
            self.get_logger().info(f'📈 平均性能: {avg_performance:.2f}, 参数样本: {self.policy.params}')
    
    def train_parameter_network(self):
        """训练参数调整网络"""
        if len(self.training_states) < 32:
            return
        
        # 随机采样训练批次
        indices = np.random.choice(len(self.training_states), 32, replace=False)
        
        total_loss = 0
        for idx in indices:
            state = self.training_states[idx]
            target_adj = self.training_adjustments[idx]
            
            # 前向传播
            predicted_adj = self.param_network.forward(state)
            
            # 计算损失
            loss = np.mean((predicted_adj - target_adj)**2)
            total_loss += loss
            
            # 反向传播
            self.param_network.backward(state, target_adj)
        
        avg_loss = total_loss / 32
        self.get_logger().info(f'🧠 参数网络训练损失: {avg_loss:.6f}')

def main():
    rclpy.init()
    
    try:
        turtle = ParameterOptimizationTurtle('turtle1')
        print("🎛️ 参数化策略优化海龟启动！")
        print("💡 神经网络将学习如何优化控制策略的参数")
        print("📊 观察策略参数如何根据性能自适应调整")
        
        rclpy.spin(turtle)
        
    except KeyboardInterrupt:
        print("\n🛑 停止优化")
    
    finally:
        if 'turtle' in locals():
            turtle.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()