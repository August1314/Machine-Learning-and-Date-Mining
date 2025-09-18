import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def monte_carlo_pi(N, num_experiments=100):
    """
    使用蒙特卡洛方法估算pi值
    
    参数:
    N: 投点总数
    num_experiments: 重复实验次数
    
    返回:
    pi_estimates: 每次实验的pi估计值
    mean_pi: 平均pi值
    variance_pi: pi值的方差
    """
    pi_estimates = []
    
    for exp in range(num_experiments):
        # 生成N个随机点，坐标在[0,1]范围内
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)
        
        # 计算距离原点的距离
        distances = np.sqrt(x**2 + y**2)
        
        # 统计落在单位圆内的点数
        points_inside_circle = np.sum(distances <= 1)
        
        # 估算pi值：pi/4 ≈ 圆内点数/总点数，所以 pi ≈ 4 * 圆内点数/总点数
        pi_estimate = 4 * points_inside_circle / N
        pi_estimates.append(pi_estimate)
    
    mean_pi = np.mean(pi_estimates)
    variance_pi = np.var(pi_estimates)
    
    return pi_estimates, mean_pi, variance_pi

def visualize_monte_carlo(N=1000, save_path=None):
    """
    可视化蒙特卡洛方法求解pi的过程
    
    参数:
    N: 投点数量
    save_path: 图片保存路径
    """
    # 生成随机点
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    
    # 判断点是否在圆内
    distances = np.sqrt(x**2 + y**2)
    inside_circle = distances <= 1
    
    # 计算pi估计值
    pi_estimate = 4 * np.sum(inside_circle) / N
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：显示所有点
    ax1.scatter(x[inside_circle], y[inside_circle], c='red', s=1, alpha=0.6, label=f'圆内点 ({np.sum(inside_circle)})')
    ax1.scatter(x[~inside_circle], y[~inside_circle], c='green', s=1, alpha=0.6, label=f'圆外点 ({np.sum(~inside_circle)})')
    
    # 绘制单位圆
    circle = Circle((0, 0), 1, fill=False, color='blue', linewidth=2, label='单位圆')
    ax1.add_patch(circle)
    
    # 绘制正方形边界
    ax1.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', linewidth=2, label='单位正方形')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'蒙特卡洛方法求解π (N={N})\nπ估计值: {pi_estimate:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：显示收敛过程
    N_values = [20, 50, 100, 200, 300, 500, 1000, 5000]
    means = []
    variances = []
    
    for n in N_values:
        _, mean_pi, var_pi = monte_carlo_pi(n, 100)
        means.append(mean_pi)
        variances.append(var_pi)
    
    ax2.errorbar(N_values, means, yerr=np.sqrt(variances), 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax2.axhline(y=np.pi, color='red', linestyle='--', linewidth=2, label=f'真实π值 ({np.pi:.4f})')
    ax2.set_xlabel('投点数量 N')
    ax2.set_ylabel('π估计值')
    ax2.set_title('不同N值下的π估计值收敛情况')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    return pi_estimate

def generate_results_table():
    """
    生成不同N值下的结果表格
    """
    N_values = [20, 50, 100, 200, 300, 500, 1000, 5000]
    results = []
    
    print("正在计算不同N值下的π估计值...")
    print("=" * 60)
    
    for N in N_values:
        pi_estimates, mean_pi, variance_pi = monte_carlo_pi(N, 100)
        std_pi = np.sqrt(variance_pi)
        
        results.append({
            'N': N,
            'Mean_π': mean_pi,
            'Variance': variance_pi,
            'Std_Dev': std_pi,
            'Error': abs(mean_pi - np.pi),
            'Relative_Error(%)': abs(mean_pi - np.pi) / np.pi * 100
        })
        
        print(f"N = {N:4d}: π = {mean_pi:.6f}, 方差 = {variance_pi:.8f}, 标准差 = {std_pi:.6f}")
    
    print("=" * 60)
    print(f"真实π值: {np.pi:.6f}")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv('/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q1/monte_carlo_results.csv', 
              index=False, encoding='utf-8-sig')
    
    print("\n结果已保存到 monte_carlo_results.csv")
    return df

def main():
    """
    主函数：执行完整的蒙特卡洛pi估算实验
    """
    print("蒙特卡洛方法求解π值")
    print("=" * 50)
    
    # 生成结果表格
    results_df = generate_results_table()
    
    # 创建可视化
    print("\n正在生成可视化图片...")
    pi_estimate = visualize_monte_carlo(N=1000, 
                                      save_path='/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q1/monte_carlo_visualization.png')
    
    # 显示结果表格
    print("\n详细结果表格:")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    return results_df

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行主程序
    results = main()

