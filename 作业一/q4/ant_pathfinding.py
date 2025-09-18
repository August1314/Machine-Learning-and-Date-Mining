import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import random

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AntPathfinding:
    def __init__(self, grid_size=7):
        """
        初始化蚂蚁路径寻找问题
        
        参数:
        grid_size: 网格大小 (n×n)
        """
        self.grid_size = grid_size
        self.start = (1, 1)
        self.end = (grid_size, grid_size)
        self.center = (4, 4)  # 中心点，可以访问0次、1次或2次
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
    def is_valid_move(self, pos, visited, center_visits):
        """
        检查移动是否有效
        
        参数:
        pos: 当前位置 (x, y)
        visited: 已访问的点集合
        center_visits: 中心点访问次数
        
        返回:
        bool: 移动是否有效
        """
        x, y = pos
        
        # 检查是否在网格边界内
        if x < 1 or x > self.grid_size or y < 1 or y > self.grid_size:
            return False
        
        # 检查是否访问过（除了中心点）
        if pos == self.center:
            return center_visits < 2  # 中心点最多访问2次
        else:
            return pos not in visited  # 其他点最多访问1次
    
    def get_valid_moves(self, pos, visited, center_visits):
        """
        获取所有有效的移动选项
        
        参数:
        pos: 当前位置 (x, y)
        visited: 已访问的点集合
        center_visits: 中心点访问次数
        
        返回:
        list: 有效的移动选项列表
        """
        valid_moves = []
        x, y = pos
        
        for dx, dy in self.directions:
            new_pos = (x + dx, y + dy)
            new_center_visits = center_visits + (1 if new_pos == self.center else 0)
            
            if self.is_valid_move(new_pos, visited, new_center_visits):
                valid_moves.append((new_pos, new_center_visits))
        
        return valid_moves
    
    def monte_carlo_simulation(self, num_simulations=20000):
        """
        蒙特卡洛仿真计算蚂蚁到达终点的概率
        
        参数:
        num_simulations: 仿真次数
        
        返回:
        success_rate: 成功到达终点的概率
        success_paths: 成功的路径列表
        """
        successful_paths = []
        successful_simulations = 0
        
        print(f"正在进行 {num_simulations} 次蒙特卡洛仿真...")
        
        for sim in range(num_simulations):
            if sim % 2000 == 0:
                print(f"已完成 {sim} 次仿真...")
            
            # 初始化
            current_pos = self.start
            visited = {self.start}
            center_visits = 1 if self.start == self.center else 0
            path = [self.start]
            
            # 随机游走
            max_steps = 100  # 防止无限循环
            step = 0
            
            while current_pos != self.end and step < max_steps:
                valid_moves = self.get_valid_moves(current_pos, visited, center_visits)
                
                if not valid_moves:
                    break  # 没有有效移动，失败
                
                # 随机选择一个有效移动
                next_pos, new_center_visits = random.choice(valid_moves)
                
                # 更新状态
                current_pos = next_pos
                visited.add(next_pos)
                center_visits = new_center_visits
                path.append(next_pos)
                step += 1
            
            # 检查是否成功到达终点
            if current_pos == self.end:
                successful_simulations += 1
                successful_paths.append(path)
        
        success_rate = successful_simulations / num_simulations
        
        print(f"仿真完成！成功次数: {successful_simulations}/{num_simulations}")
        print(f"成功概率: {success_rate:.6f}")
        
        return success_rate, successful_paths
    
    def find_all_paths_dfs(self):
        """
        使用深度优先搜索找到所有可能的路径（用于小规模验证）
        """
        all_paths = []
        
        def dfs(current_pos, visited, center_visits, path):
            if current_pos == self.end:
                all_paths.append(path[:])
                return
            
            valid_moves = self.get_valid_moves(current_pos, visited, center_visits)
            
            for next_pos, new_center_visits in valid_moves:
                new_visited = visited.copy()
                new_visited.add(next_pos)
                new_path = path + [next_pos]
                
                dfs(next_pos, new_visited, new_center_visits, new_path)
        
        # 开始DFS搜索
        dfs(self.start, {self.start}, 1 if self.start == self.center else 0, [self.start])
        
        return all_paths
    
    def visualize_grid_and_paths(self, successful_paths=None, save_path=None):
        """
        可视化网格和路径
        
        参数:
        successful_paths: 成功的路径列表
        save_path: 图片保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：网格和中心点
        ax1.set_xlim(0.5, self.grid_size + 0.5)
        ax1.set_ylim(0.5, self.grid_size + 0.5)
        ax1.set_aspect('equal')
        
        # 绘制网格
        for i in range(1, self.grid_size + 1):
            for j in range(1, self.grid_size + 1):
                rect = plt.Rectangle((i-0.5, j-0.5), 1, 1, 
                                   fill=False, edgecolor='black', linewidth=0.5)
                ax1.add_patch(rect)
        
        # 标记起点和终点
        ax1.plot(self.start[0], self.start[1], 'go', markersize=10, label='起点 A(1,1)')
        ax1.plot(self.end[0], self.end[1], 'ro', markersize=10, label='终点 B(7,7)')
        
        # 标记中心点
        ax1.plot(self.center[0], self.center[1], 'bo', markersize=8, label='中心点(4,4)')
        
        # 绘制成功路径（如果提供）
        if successful_paths and len(successful_paths) > 0:
            # 随机选择几条路径进行展示
            sample_paths = random.sample(successful_paths, min(5, len(successful_paths)))
            
            for i, path in enumerate(sample_paths):
                x_coords = [pos[0] for pos in path]
                y_coords = [pos[1] for pos in path]
                ax1.plot(x_coords, y_coords, 'r-', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{self.grid_size}×{self.grid_size} 网格和蚂蚁路径')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：路径长度分布
        if successful_paths:
            path_lengths = [len(path) for path in successful_paths]
            
            ax2.hist(path_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel('路径长度')
            ax2.set_ylabel('频次')
            ax2.set_title('成功路径长度分布')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_length = np.mean(path_lengths)
            std_length = np.std(path_lengths)
            ax2.axvline(mean_length, color='red', linestyle='--', 
                       label=f'平均长度: {mean_length:.2f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, '无成功路径', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title('路径长度分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
    
    def analyze_paths(self, successful_paths):
        """
        分析成功路径的特征
        
        参数:
        successful_paths: 成功的路径列表
        """
        if not successful_paths:
            print("没有成功路径进行分析")
            return
        
        print("\n路径分析:")
        print("=" * 50)
        
        # 路径长度统计
        path_lengths = [len(path) for path in successful_paths]
        print(f"总成功路径数: {len(successful_paths)}")
        print(f"最短路径长度: {min(path_lengths)}")
        print(f"最长路径长度: {max(path_lengths)}")
        print(f"平均路径长度: {np.mean(path_lengths):.2f}")
        print(f"路径长度标准差: {np.std(path_lengths):.2f}")
        
        # 中心点访问统计
        center_visits = []
        for path in successful_paths:
            visits = path.count(self.center)
            center_visits.append(visits)
        
        print(f"\n中心点访问统计:")
        print(f"平均访问次数: {np.mean(center_visits):.2f}")
        print(f"访问0次: {center_visits.count(0)} 条路径")
        print(f"访问1次: {center_visits.count(1)} 条路径")
        print(f"访问2次: {center_visits.count(2)} 条路径")
        
        # 方向偏好分析
        direction_counts = {'右': 0, '下': 0, '左': 0, '上': 0}
        direction_names = ['右', '下', '左', '上']
        
        for path in successful_paths:
            for i in range(len(path) - 1):
                current = path[i]
                next_pos = path[i + 1]
                dx = next_pos[0] - current[0]
                dy = next_pos[1] - current[1]
                
                if dx == 1 and dy == 0:
                    direction_counts['右'] += 1
                elif dx == 0 and dy == 1:
                    direction_counts['下'] += 1
                elif dx == -1 and dy == 0:
                    direction_counts['左'] += 1
                elif dx == 0 and dy == -1:
                    direction_counts['上'] += 1
        
        print(f"\n移动方向统计:")
        total_moves = sum(direction_counts.values())
        for direction, count in direction_counts.items():
            percentage = count / total_moves * 100 if total_moves > 0 else 0
            print(f"{direction}: {count} 次 ({percentage:.1f}%)")

def main():
    """
    主函数：执行蚂蚁路径寻找的蒙特卡洛仿真
    """
    print("蚂蚁路径寻找问题 - 蒙特卡洛仿真")
    print("=" * 50)
    print("问题描述:")
    print("- 蚂蚁从(1,1)出发，目标到达(7,7)")
    print("- 只能向右、向下、向左、向上移动")
    print("- 除中心点(4,4)外，每个点最多访问1次")
    print("- 中心点(4,4)最多访问2次")
    print("=" * 50)
    
    # 创建蚂蚁路径寻找实例
    ant = AntPathfinding(grid_size=7)
    
    # 进行蒙特卡洛仿真
    success_rate, successful_paths = ant.monte_carlo_simulation(num_simulations=20000)
    
    # 分析路径
    ant.analyze_paths(successful_paths)
    
    # 可视化结果
    print("\n正在生成可视化图片...")
    ant.visualize_grid_and_paths(successful_paths, 
                                save_path='/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q4/ant_pathfinding_visualization.png')
    
    # 保存结果
    results = {
        'Grid_Size': 7,
        'Success_Rate': success_rate,
        'Total_Simulations': 20000,
        'Successful_Paths': len(successful_paths),
        'Start_Point': '(1,1)',
        'End_Point': '(7,7)',
        'Center_Point': '(4,4)',
        'Max_Center_Visits': 2
    }
    
    df = pd.DataFrame([results])
    df.to_csv('/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q4/ant_pathfinding_results.csv', 
              index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到 ant_pathfinding_results.csv")
    print(f"成功概率 P = {success_rate:.6f}")
    
    # 提供选择题答案
    print("\n选择题答案选项分析:")
    print("=" * 30)
    options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    closest_option = min(options, key=lambda x: abs(x - success_rate))
    print(f"最接近的选项: {closest_option}")
    print(f"实际概率: {success_rate:.6f}")
    print(f"误差: {abs(success_rate - closest_option):.6f}")
    
    return success_rate, successful_paths

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 运行主程序
    success_rate, paths = main()

