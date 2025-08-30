import numpy as np
import random

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
        
        sample_size = min(30, len(data))
        sampled_data = random.sample(data, sample_size)
        for params, lap_time in sampled_data:
            predicted = self.forward(params)
            loss = self.backward(np.array([lap_time]))
            total_loss += loss
            sample_count += 1
            
            if logger:
                logger.info(f"预测:{predicted[0]:.1f}s, 实际:{lap_time:.1f}s, 误差:{abs(predicted[0]-lap_time):.1f}s")
    