#!/bin/bash

# 启动turtlesim
echo "启动 turtlesim..."
ros2 run turtlesim turtlesim_node --ros-args -p width:=1800 -p height:=1600 --log-level ERROR 2>/dev/null &
sleep 2

# 选择路线
echo "选择路线:"
echo "1) simple      2) figure_eight  3) spiral      4) star"
echo "5) complex_polygon  6) s_curve  7) maze        8) diamond"
echo "9) triangle    10) zigzag"
read -p "选择 (1-10, 默认1): " route_choice

case $route_choice in
    1|"") ROUTE="simple" ;;
    2) ROUTE="figure_eight" ;;
    3) ROUTE="spiral" ;;
    4) ROUTE="star" ;;
    5) ROUTE="complex_polygon" ;;
    6) ROUTE="s_curve" ;;
    7) ROUTE="maze" ;;
    8) ROUTE="diamond" ;;
    9) ROUTE="triangle" ;;
    10) ROUTE="zigzag" ;;
    *) ROUTE="simple" ;;
esac

# 选择是否继续之前的参数
read -p "继续之前的最佳参数? (y/N): " continue_choice
if [[ $continue_choice =~ ^[Yy]$ ]]; then
    CONTINUE="--contin"
    message="从训练后的参数开始"
else
    CONTINUE=""
    message="从初始开始"
fi

# 启动控制器
echo "启动 AI控制器: 路线=$ROUTE, $message"
python3 src/turtle_ai_controller.py --route $ROUTE $CONTINUE