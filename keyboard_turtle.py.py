#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys

class KeyboardTurtle(Node):
    def __init__(self):
        super().__init__('keyboard_turtle')
        
        # 发布移动命令
        self.cmd_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        self.get_logger().info('键盘控制启动！')
        self.print_help()
        
        # 开始接收键盘输入
        self.get_user_input()

    def print_help(self):
        print("""
        简单控制:
        w - 前进    s - 后退
        a - 左转    d - 右转
        空格 - 停止  q - 退出
        """)

    def get_user_input(self):
        """获取用户输入"""
        while rclpy.ok():
            try:
                key = input("输入命令 (w/s/a/d/空格/q): ").strip().lower()
                
                if key == 'q':
                    break
                elif key == 'w':
                    self.move_forward()
                elif key == 's':
                    self.move_backward() 
                elif key == 'a':
                    self.turn_left()
                elif key == 'd':
                    self.turn_right()
                elif key == '' or key == ' ':
                    self.stop()
                else:
                    print("无效输入！")
                    
            except KeyboardInterrupt:
                break

    def move_forward(self):
        twist = Twist()
        twist.linear.x = 2.0
        self.cmd_publisher.publish(twist)
        print("前进！")

    def move_backward(self):
        twist = Twist()
        twist.linear.x = -2.0
        self.cmd_publisher.publish(twist)
        print("后退！")

    def turn_left(self):
        twist = Twist()
        twist.angular.z = 2.0
        self.cmd_publisher.publish(twist)
        print("左转！")

    def turn_right(self):
        twist = Twist()
        twist.angular.z = -2.0
        self.cmd_publisher.publish(twist)
        print("右转！")

    def stop(self):
        twist = Twist()  # 全为0
        self.cmd_publisher.publish(twist)
        print("停止！")

def main():
    rclpy.init()
    node = KeyboardTurtle()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()