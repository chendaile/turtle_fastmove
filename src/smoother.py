#!/usr/bin/env python3
import json
import argparse
import sys
import os

def interpolate_points(start, end, num_points=3):
    """在两点之间插值生成中间点"""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append([round(x, 2), round(y, 2)])
    return points

def interpolate_route(route, points_between=1):
    """给路线的每条边插入中间点"""
    if not route:
        return route
    
    new_route = {}
    new_index = 0
    
    # 获取所有点，按索引排序
    sorted_points = sorted([(int(k), v) for k, v in route.items()])
    
    for i in range(len(sorted_points)):
        current_point = sorted_points[i][1]
        next_point = sorted_points[(i + 1) % len(sorted_points)][1]
        
        # 在当前点和下一个点之间插值
        interpolated = interpolate_points(current_point, next_point, points_between + 2)
        
        # 添加插值点（除了最后一个点）
        for j in range(len(interpolated) - 1):
            new_route[new_index] = interpolated[j]
            new_index += 1
    
    return new_route

def process_routes_file(input_file, output_file, points_between):
    """处理路线文件，保持原有JSON结构"""
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        return False
    
    try:
        # 读取原始JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查JSON结构
        if 'routes' not in data:
            print("错误: JSON文件中没有找到 'routes' 字段")
            return False
        
        # 处理每个路线
        new_data = data.copy()  # 保持所有原有字段
        new_routes = {}
        
        for route_name, route in data['routes'].items():
            original_count = len(route)
            interpolated_route = interpolate_route(route, points_between)
            new_count = len(interpolated_route)
            
            new_routes[route_name] = interpolated_route
            print(f"{route_name}: {original_count} -> {new_count} 个点")
        
        # 更新routes部分，保持其他字段不变
        new_data['routes'] = new_routes
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n插值完成! 输出文件: {output_file}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式不正确 - {e}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='路线插值工具 - 在路线点之间插入中间点使路径更平滑',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python route_interpolator.py config/routes.json -o config/routes_smooth.json -n 2
  python route_interpolator.py routes.json -n 1
  python route_interpolator.py routes.json --points 3 --output smooth_routes.json
        '''
    )
    
    parser.add_argument('input', help='输入的路线JSON文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径 (默认: 输入文件名_smooth.json)')
    parser.add_argument('-n', '--points', type=int, default=1, 
                       help='每两个原始点之间插入的点数 (默认: 1)')
    
    args = parser.parse_args()
    
    # 设置默认输出文件名
    if not args.output:
        input_name = os.path.splitext(args.input)[0]
        args.output = f"{input_name}_smooth.json"
    
    # 验证参数
    if args.points < 1:
        print("错误: 插入点数必须大于等于1")
        sys.exit(1)
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"每段插入点数: {args.points}")
    print("-" * 40)
    
    # 处理文件
    success = process_routes_file(args.input, args.output, args.points)
    
    if success:
        print(f"\n成功生成平滑路线文件: {args.output}")
        sys.exit(0)
    else:
        print("\n处理失败")
        sys.exit(1)

if __name__ == "__main__":
    main()