import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')  # 忽略sklearn的收敛警告

class network:
    def __init__(self, layers=None, activation='relu', learning_rate=0.001):
        # sklearn的MLPRegressor只需要隐藏层尺寸，不包括输入输出层
        if layers is None:
            hidden_layers = (16, 8)  # 默认两个隐藏层
        else:
            # 去掉输入层和输出层，只保留隐藏层
            hidden_layers = tuple(layers[1:-1]) if len(layers) > 2 else (8,)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,   # 隐藏层结构
            activation=activation,              # 激活函数: 'relu', 'tanh', 'logistic'
            learning_rate_init=learning_rate,   # 初始学习率
            learning_rate='constant',           # 学习率策略
            max_iter=50,                       # 每次训练最大迭代次数
            warm_start=True,                   # 保持之前的训练结果
            random_state=42,                   # 随机种子
            alpha=0.0001,                      # L2正则化参数
            solver='adam',                     # 优化器
            early_stopping=False,              # 不使用早停
            validation_fraction=0.1,           # 验证集比例
            n_iter_no_change=10,               # 早停的容忍轮数
            tol=1e-4                           # 收敛容忍度
        )
        
        # 数据标准化器
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.is_fitted = False
        self.training_data = []
    
    def forward(self, input_data):
        """前向传播 - 预测"""
        if not self.is_fitted:
            # 如果模型还没训练，返回一个合理的默认预测
            return np.array([15.0])  # 默认15秒
        
        try:
            # 标准化输入
            input_scaled = self.scaler_X.transform(input_data.reshape(1, -1))
            
            # 预测
            prediction_scaled = self.model.predict(input_scaled)
            
            # 反标准化输出
            prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(1, -1))
            
            # 确保预测值在合理范围内
            prediction = np.clip(prediction[0], 1.0, 300.0)
            
            return prediction
        except Exception as e:
            print(f"预测错误: {e}")
            return np.array([15.0])
    
    def backward(self, target_output):
        """后向传播 - sklearn会自动处理，这里只是为了兼容接口"""
        # sklearn的MLPRegressor会在fit时自动进行反向传播
        # 这里返回一个虚拟的损失值
        if self.is_fitted:
            try:
                # 计算当前预测的损失
                X = np.array([data[0] for data in self.training_data[-10:]])
                y = np.array([data[1] for data in self.training_data[-10:]])
                
                if len(X) > 0:
                    X_scaled = self.scaler_X.transform(X)
                    y_pred_scaled = self.model.predict(X_scaled)
                    y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    mse = np.mean((y_pred - y) ** 2)
                    return float(mse)
            except:
                pass
        
        return 100.0  # 默认损失值
    
    def train_network(self, data, logger=None):
        """训练网络"""
        if len(data) < 3:
            return float('inf')
        
        try:
            # 准备训练数据
            sample_size = min(50, len(data))  # 使用更多数据训练
            sampled_data = random.sample(data, sample_size)
            
            X = np.array([params for params, _ in sampled_data])
            y = np.array([lap_time for _, lap_time in sampled_data])
            
            # 数据标准化
            if not self.is_fitted:
                # 第一次训练，拟合标准化器
                X_scaled = self.scaler_X.fit_transform(X)
                y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                # 后续训练，使用已有的标准化器
                X_scaled = self.scaler_X.transform(X)
                y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            
            # 训练模型
            self.model.fit(X_scaled, y_scaled)
            self.is_fitted = True
            self.training_data = data.copy()  # 保存训练数据
            
            # 计算训练损失
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            mse = np.mean((y_pred - y) ** 2)
            
            # 显示一些预测结果
            if logger:
                for i, (pred, actual) in enumerate(zip(y_pred[:3], y[:3])):
                    error = abs(pred - actual)
                    logger.info(f"预测:{pred:.1f}s, 实际:{actual:.1f}s, 误差:{error:.1f}s")
                
                logger.info(f"平均训练损失: {mse:.3f}")
            
            return mse
            
        except Exception as e:
            if logger:
                logger.info(f"训练失败: {e}")
            return float('inf')