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

        self.current_paramList = np.array(list(self.params.values()))

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
        self.layer01_size = (9, 8)
        self.layer02_size = (8, 1)

        self.weight01 = np.random.rand(*self.layer01_size) * 0.1
        self.bias01 = np.random.rand(self.layer01_size[1]) * 0.01
        self.weight02 = np.random.rand(*self.layer02_size) * 0.1
        self.bias02 = np.random.rand(self.layer02_size[1]) * 0.01

        self.learning_rate = 0.001
        
        # 添加训练历史记录
        self.training_losses = []
        self.prediction_errors = []

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, input_data: np.ndarray):
        self.input_data = input_data
        self.z1 = input_data @ self.weight01 + self.bias01
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.weight02 + self.bias02
        self.a2 = self.tanh(self.z2)

        return self.a2
    
    def backward(self, target_output: np.ndarray):
        output_error = self.a2 - target_output
        delta2 = output_error * self.tanh_derivative(self.z2)
        
        error1 = delta2 @ self.weight02.T
        delta1 = error1 * self.relu_derivative(self.z1)
        
        self.weight02 -= self.learning_rate * np.outer(self.a1, delta2)
        self.bias02 -= self.learning_rate * delta2
        
        self.weight01 -= self.learning_rate * np.outer(self.input_data, delta1)
        self.bias01 -= self.learning_rate * delta1
        
        # 计算并记录损失
        loss = np.mean(output_error**2)  # MSE损失
        self.training_losses.append(float(loss))
        
        return loss

    def train_network(self, data, logger=None):
        """训练网络并返回训练统计信息"""
        batch_losses = []
        predictions_vs_actual = []
        
        training_samples = data[-5:]  # 使用最近5次数据
        
        for params, actual_lap_time in training_samples:
            # 前向传播
            predicted_time = self.forward(params)
            
            # 记录预测vs实际
            predictions_vs_actual.append({
                'predicted': float(predicted_time[0]),
                'actual': float(actual_lap_time),
                'error': abs(float(predicted_time[0]) - float(actual_lap_time))
            })
            
            # 反向传播
            loss = self.backward(np.array([actual_lap_time]))
            batch_losses.append(loss)
        
        # 计算统计信息
        avg_loss = np.mean(batch_losses)
        avg_error = np.mean([item['error'] for item in predictions_vs_actual])
        
        # 记录预测误差历史
        self.prediction_errors.append(avg_error)
        
        # 日志输出
        if logger:
            logger.info(f"=== 神经网络训练统计 ===")
            logger.info(f"训练样本数: {len(training_samples)}")
            logger.info(f"平均损失: {avg_loss:.4f}")
            logger.info(f"平均预测误差: {avg_error:.2f}秒")
            
            # 详细的预测对比
            for i, item in enumerate(predictions_vs_actual):
                logger.info(f"样本{i+1}: 预测={item['predicted']:.2f}s, "
                          f"实际={item['actual']:.2f}s, "
                          f"误差={item['error']:.2f}s")
            
            # 学习趋势分析
            if len(self.prediction_errors) >= 3:
                recent_errors = self.prediction_errors[-3:]
                if recent_errors[-1] < recent_errors[0]:
                    logger.info("趋势: 预测准确度在提升 ↗")
                elif recent_errors[-1] > recent_errors[0]:
                    logger.info("趋势: 预测准确度在下降 ↘")
                else:
                    logger.info("趋势: 预测准确度保持稳定 →")
        
        return {
            'avg_loss': avg_loss,
            'avg_error': avg_error,
            'predictions': predictions_vs_actual,
            'is_improving': len(self.prediction_errors) >= 2 and self.prediction_errors[-1] < self.prediction_errors[-2]
        }
    
    def evaluate_fitness_quality(self, data, logger=None):
        """评估网络的拟合质量"""
        if len(data) < 3:
            return None
            
        # 使用所有历史数据进行评估
        all_predictions = []
        all_actuals = []
        
        for params, actual_time in data:
            predicted = self.forward(params)[0]
            all_predictions.append(float(predicted))
            all_actuals.append(float(actual_time))
        
        # 计算拟合指标
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        
        # R²决定系数
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 平均绝对误差
        mae = np.mean(np.abs(actuals - predictions))
        
        # 相关系数
        correlation = np.corrcoef(predictions, actuals)[0,1] if len(actuals) > 1 else 0
        
        if logger:
            logger.info(f"=== 拟合质量评估 ===")
            logger.info(f"R²决定系数: {r2_score:.4f} (越接近1越好)")
            logger.info(f"平均绝对误差: {mae:.2f}秒")
            logger.info(f"相关系数: {correlation:.4f} (越接近1越好)")
            
            # 拟合质量等级
            if r2_score > 0.8:
                logger.info("拟合质量: 优秀")
            elif r2_score > 0.6:
                logger.info("拟合质量: 良好") 
            elif r2_score > 0.4:
                logger.info("拟合质量: 一般")
            else:
                logger.info("拟合质量: 较差，需要更多数据")
        
        return {
            'r2_score': r2_score,
            'mae': mae,
            'correlation': correlation,
            'quality_level': 'excellent' if r2_score > 0.8 else 
                           'good' if r2_score > 0.6 else 
                           'fair' if r2_score > 0.4 else 'poor'
        }
    
class turtle_node(Node):
    def __init__(self):
        super().__init__('turtle1')
        self.param = optimized_para()
        self.brain = network()
        self.get_logger().info("Start turtle node!")

        # ... 其他初始化代码保持不变 ...
        
        # 添加优化历史记录
        self.optimization_history = []
        self.best_lap_time = float('inf')

    # ... 其他方法保持不变 ...

    def optimize_parameters(self):
        """优化参数并记录详细日志"""
        self.get_logger().info("=== 开始参数优化 ===")
        
        best_params = None
        best_predicted_time = float('inf')
        optimization_candidates = []
        
        for i in range(10): 
            candidate_params = self.param.generate_candidate_params()
            predicted_time = self.brain.forward(candidate_params)[0]
            
            optimization_candidates.append({
                'candidate': i+1,
                'predicted_time': float(predicted_time),
                'params': candidate_params.copy()
            })
            
            if predicted_time < best_predicted_time:
                best_predicted_time = predicted_time
                best_params = candidate_params
        
        # 记录优化过程
        self.get_logger().info(f"测试了10个候选参数组合:")
        sorted_candidates = sorted(optimization_candidates, key=lambda x: x['predicted_time'])
        
        for i, candidate in enumerate(sorted_candidates[:3]):  # 显示前3个最佳
            self.get_logger().info(f"第{i+1}佳: 预测时间 {candidate['predicted_time']:.2f}秒")
        
        if best_params is not None:
            old_params = self.param.current_paramList.copy()
            self.param.update_params(best_params)
            
            # 计算参数变化
            param_changes = np.abs(best_params - old_params)
            max_change_idx = np.argmax(param_changes)
            param_names = list(self.param.params.keys())
            
            self.get_logger().info(f"采用最优参数组合! 预测改进时间: {best_predicted_time:.2f}秒")
            self.get_logger().info(f"最大参数变化: {param_names[max_change_idx]} "
                                 f"({old_params[max_change_idx]:.3f} → {best_params[max_change_idx]:.3f})")
            
            # 记录优化历史
            self.optimization_history.append({
                'lap_count': len(self.training_data),
                'predicted_improvement': float(best_predicted_time),
                'param_changes': param_changes,
                'max_change_param': param_names[max_change_idx]
            })

    def print_status(self):
        """增强的状态打印，包含学习效果分析"""
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"总路程: {self.total_route:.2f}m")
        self.get_logger().info(f"总时间: {self.total_time:.2f}s") 
        self.get_logger().info(f"平均速度: {self.total_route / self.total_time:.2f}m/s")
        self.get_logger().info(f"完成节点: {self.total_node}")
        
        # 当前圈时间分析
        if len(self.training_data) > 0:
            current_lap_time = self.training_data[-1][1]
            self.get_logger().info(f"当前圈时间: {current_lap_time:.2f}秒")
            
            if current_lap_time < self.best_lap_time:
                self.best_lap_time = current_lap_time
                self.get_logger().info(f"新记录! 最佳圈时间: {self.best_lap_time:.2f}秒")
            else:
                self.get_logger().info(f"历史最佳: {self.best_lap_time:.2f}秒")
        
        # 学习进度分析
        if len(self.training_data) >= 3:
            recent_times = [data[1] for data in self.training_data[-3:]]
            if recent_times[-1] < recent_times[0]:
                self.get_logger().info("性能趋势: 持续改进中")
            elif recent_times[-1] > recent_times[0]:
                self.get_logger().info("性能趋势: 需要调整策略")
            
        # 当前优化参数
        self.get_logger().info("当前参数:")
        for key, value in self.param.params.items():
            self.get_logger().info(f"  {key}: {value:.3f}")
        
        self.get_logger().info("=" * 50)

    def mainloop(self):
        if self.total_node > 0 and self.total_node % 4 == 0 and not self.havePrint:
            self.havePrint = True
            lap_time = time.time() - self.lap_start_time
            self.training_data.append((self.param.current_paramList.copy(), lap_time))
            
            self.get_logger().info(f"完成第{len(self.training_data)}圈，用时: {lap_time:.2f}秒")
            
            if len(self.training_data) > 2:  # 至少3次数据
                # 训练网络并获取统计信息
                train_stats = self.brain.train_network(self.training_data, self.get_logger())
                
                # 评估拟合质量
                fitness_stats = self.brain.evaluate_fitness_quality(self.training_data, self.get_logger())
                
                # 只有在拟合质量较好时才进行参数优化
                if fitness_stats and fitness_stats['r2_score'] > 0.3:
                    self.optimize_parameters()
                else:
                    self.get_logger().info("拟合质量较差，继续收集数据...")
            
            self.lap_start_time = time.time()
            self.print_status()
            
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
