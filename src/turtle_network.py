import numpy as np
import random

class network:
    def __init__(self, layers=[9, 8, 4, 1], activation='relu', learning_rate=0.005):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(len(layers)-1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # 存储前向传播的值
        self.activations = []
        self.z_values = []
    
    def activate(self, x):
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return x
    
    def activate_derivative(self, x):
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation == 'sigmoid':
            s = self.activate(x)
            return s * (1 - s)
        return np.ones_like(x)
    
    def forward(self, input_data):
        self.activations = [input_data]
        self.z_values = []
        
        current = input_data
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:  # 隐藏层用激活函数
                current = self.activate(z)
            else:  # 输出层线性
                current = z  # 保持原来的线性输出
            
            self.activations.append(current)
        
        return current
    
    def backward(self, target_output):
        # 输出层误差
        output_error = self.activations[-1] - target_output
        delta = output_error
        
        # 反向传播
        for i in range(len(self.weights)-1, -1, -1):
            # 更新权重和偏置
            grad_w = np.outer(self.activations[i], delta)
            grad_b = delta
            
            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b
            
            # 计算下一层的误差（除了第一层）
            if i > 0:
                error = delta @ self.weights[i].T
                delta = error * self.activate_derivative(self.z_values[i-1])
        
        return float(output_error[0] ** 2)

    def train_network(self, data, logger=None):
        total_loss = 0
        sample_count = 0
        diffs = []
        
        sample_size = min(30, len(data))
        sampled_data = random.sample(data, sample_size)
        for params, lap_time in sampled_data:
            predicted = self.forward(params)
            loss = self.backward(np.array([lap_time]))
            total_loss += loss
            sample_count += 1
            diff = abs(predicted[0]-lap_time)
            diffs.append(diff)
            
            if logger and sample_count <= 3:  # 只显示前5个结果，避免日志过多
                logger.info(f"预测:{predicted[0]:.1f}s, 实际:{lap_time:.1f}s, 误差:{diff:.1f}s")
        diffs = np.array(diffs)
        logger.info(f"平均差异: {np.mean(diffs):.2f}s")
    