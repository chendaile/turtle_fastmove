ros2 run turtlesim turtlesim_node --ros-args -p width:=1800 -p height:=1600 &
sleep 2  # 等待turtlesim启动
python3 src/turtle_ai_controller.py