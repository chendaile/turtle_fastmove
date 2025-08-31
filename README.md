# Turtle AI Controller / 智能海龟控制系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ROS2](https://img.shields.io/badge/ROS2-Humble+-green.svg)](https://docs.ros.org/en/humble/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An intelligent turtle control system based on ROS2 that uses neural networks to optimize path-following parameters in real-time.*

基于ROS2的智能海龟控制系统，通过神经网络参数优化实现自适应路径跟踪。

## 🚀 Quick Start / 快速开始

### Interactive Launch / 交互式启动
```bash
./start.sh
```
Follow the prompts to select route and parameter configuration.  
根据提示选择路线和参数配置即可开始训练。

### Direct Launch / 直接启动
```bash
# Start turtlesim first / 首先启动turtlesim
ros2 run turtlesim turtlesim_node --ros-args -p width:=1800 -p height:=1600 --log-level ERROR &

# Run with specific route / 运行特定路线
python3 src/turtle_ai_controller.py --route spiral --contin
```

## 📖 Project Overview / 项目简介

This project implements an intelligent turtle control system where:
- A neural network learns the mapping between control parameters and lap times in real-time
- The system automatically optimizes turning speed, angle threshold, and other control parameters
- Best parameters are updated after each lap completion, continuously improving performance

本项目实现了一个智能海龟控制系统，海龟沿预设路线行走的过程中：
- 神经网络实时学习控制参数与圈速的映射关系
- 系统自动优化转向速度、角度阈值等控制参数
- 每完成一圈后更新最佳参数，持续提升行走效率

## ✨ Core Features / 核心特性

- **🛣️ 10 Predefined Routes**: From simple squares to complex mazes, adapting to different training needs  
  **10种预设路线**：从简单方形到复杂迷宫，适应不同训练需求

- **🧠 Real-time Parameter Optimization**: Intelligent parameter adjustment based on neural network predictions  
  **实时参数优化**：基于神经网络预测的智能参数调整

- **💾 Best Record Persistence**: Automatically saves historical optimal parameters, supports continuous training  
  **最佳记录保存**：自动保存历史最优参数，支持继续训练

- **📊 Visual Feedback**: Real-time display of lap times, parameter changes, and training progress  
  **可视化反馈**：实时显示圈速、参数变化和训练进度

## 🏗️ System Architecture / 系统架构

### Control Parameters (9 total) / 控制参数（9个）

| Parameter | Range | Description |
|-----------|-------|-------------|
| `angle_threshold` | [0.02, 3.0] | Angular precision threshold / 角度精度阈值 |
| `turn_speed` | [1.0, 10.0] | Base turning speed / 基础转向速度 |
| `speed_angle_factor` | [0.5, 5.0] | Speed adjustment based on angle error / 基于角度误差的速度调整 |
| `min_turn_speed` | [0.01, 1.0] | Minimum turning speed / 最小转向速度 |
| `fine_tune_gain` | [0.05, 5.0] | Fine adjustment gain / 微调增益 |
| `speed_distance_factor` | [0.5, 10.0] | Speed adjustment based on distance / 基于距离的速度调整 |
| `max_speed` | [1.0, 5.0] | Maximum forward speed / 最大前进速度 |
| `slow_distance` | [0.1, 5.0] | Distance threshold for slowing down / 减速距离阈值 |
| `slow_factor` | [0.1, 10.0] | Speed reduction factor when slowing / 减速时的速度减少因子 |

### Neural Network (4 layers) / 神经网络（4层）
- **Input**: 9 control parameters / **输入**：9个控制参数
- **Hidden layers**: 16 → 8 neurons / **隐藏层**：16 → 8 全连接层
- **Output**: Predicted lap completion time / **输出**：预测完成一圈所需时间

### Optimization Strategy / 优化策略
1. Collect parameter-laptime data pairs / 收集参数-圈速数据对
2. Train neural network to establish mapping / 训练神经网络建立映射关系
3. Generate candidate parameters and predict performance / 生成候选参数并预测性能
4. Select optimal parameters to update controller / 选择最优参数更新控制器

## 🛤️ Available Routes / 可用路线

| Route | Points | Difficulty | Description |
|-------|--------|------------|-------------|
| `simple` | 4 | ⭐ | Basic square route / 基础方形路线 |
| `triangle` | 3 | ⭐ | Minimal point test / 最少点数测试 |
| `diamond` | 4 | ⭐⭐ | Diagonal navigation / 对角线导航 |
| `figure_eight` | 8 | ⭐⭐ | Continuous turning test / 连续转向测试 |
| `spiral` | 10 | ⭐⭐ | Progressive navigation / 渐进式导航 |
| `zigzag` | 6 | ⭐⭐ | Frequent direction changes / 频繁方向切换 |
| `s_curve` | 10 | ⭐⭐⭐ | Smooth turning test / 平滑转向测试 |
| `star` | 10 | ⭐⭐⭐ | Precise geometric control / 精确几何控制 |
| `complex_polygon` | 9 | ⭐⭐⭐ | Comprehensive performance test / 综合性能测试 |
| `maze` | 13 | ⭐⭐⭐⭐ | Ultimate difficulty challenge / 最高难度挑战 |

## 💻 Command Line Usage / 命令行使用

```bash
# Run with specific route and continue from best parameters
# 使用特定路线并从最佳参数继续
python3 src/turtle_ai_controller.py --route spiral --contin

# Available options / 可用选项
--route, -r    Select route (default: simple) / 选择路线（默认：simple）
--contin, -u   Continue from best saved parameters / 从历史最佳参数开始
```

### Available routes / 可用路线名称
`simple`, `figure_eight`, `spiral`, `star`, `complex_polygon`, `s_curve`, `maze`, `diamond`, `triangle`, `zigzag`

## 📁 Project Structure / 文件结构

```
turtle_fastmove/
├── src/                           # Source code / 源代码
│   ├── turtle_ai_controller.py    # Main ROS2 controller / 主控制器
│   ├── turtle_network_sci.py      # Neural network implementation / 神经网络实现
│   └── turtle_para.py             # Parameter management / 参数管理
├── config/                        # Configuration files / 配置文件
│   ├── init_arg.json              # Initial parameters and ranges / 初始参数和范围
│   └── routes.json                # Route coordinate definitions / 路线坐标定义
├── output/                        # Best parameters storage / 最佳参数存储
│   └── best_params_*.json         # Per-route best parameters / 各路线最佳参数
├── turtle_env/                    # Python virtual environment / Python虚拟环境
├── start.sh                       # Interactive launch script / 交互式启动脚本
├── README.md                      # This file / 本文件
└── CLAUDE.md                      # Project instructions for Claude Code
```

## 🔧 Requirements / 运行要求

### System Dependencies / 系统依赖
- **ROS2**: Humble or newer / Humble或更新版本
- **Python**: 3.8+ 
- **Display**: 1800x1600 pixels recommended / 建议1800x1600像素显示空间

### Python Packages / Python包
```bash
pip install numpy scikit-learn rclpy geometry-msgs
```

### Setup / 设置
```bash
# Install ROS2 Humble (Ubuntu 22.04) / 安装ROS2 Humble
sudo apt update && sudo apt install ros-humble-desktop

# Source ROS2 / 配置ROS2环境
source /opt/ros/humble/setup.bash

# Install turtlesim / 安装turtlesim
sudo apt install ros-humble-turtlesim

# Clone and run / 克隆并运行
git clone <repository-url>
cd turtle_fastmove
chmod +x start.sh
./start.sh
```

## 🎯 Training Process / 训练过程

1. **Parameter Initialization**: Load from `config/init_arg.json` or best saved parameters  
   **参数初始化**：从 `config/init_arg.json` 或最佳保存参数加载

2. **Route Execution**: Turtle follows selected route using current parameters  
   **路线执行**：海龟使用当前参数跟随选定路线

3. **Performance Evaluation**: Record lap completion time  
   **性能评估**：记录完成一圈的时间

4. **Neural Network Training**: Update model with new parameter-time data pair  
   **神经网络训练**：使用新的参数-时间数据对更新模型

5. **Parameter Optimization**: Generate and evaluate candidate parameters  
   **参数优化**：生成并评估候选参数

6. **Best Parameter Update**: Save improved parameters to `output/`  
   **最佳参数更新**：将改进的参数保存到 `output/` 目录

## 📊 Performance Monitoring / 性能监控

The system provides real-time feedback including:
- Current lap time / 当前圈速
- Best recorded time / 历史最佳时间  
- Parameter values / 参数值
- Training iteration count / 训练迭代次数
- Neural network prediction accuracy / 神经网络预测精度

## 🤝 Contributing / 贡献

Contributions are welcome! Please feel free to submit issues and pull requests.  
欢迎贡献！请随时提交问题和拉取请求。

## 📄 License / 许可证

This project is licensed under the MIT License.  
本项目采用MIT许可证。