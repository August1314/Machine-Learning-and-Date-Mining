import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def monte_carlo_double_integration(N, num_experiments=100):
    """
    使用蒙特卡洛方法计算二重积分 ∫₀¹ ∫₀¹ e^(-(x²+y²)) dx dy
    
    参数:
    N: 采样点数
    num_experiments: 重复实验次数
    
    返回:
    integral_estimates: 每次实验的积分估计值
    mean_integral: 平均积分值
    variance_integral: 积分值的方差
    """
    integral_estimates = []
    
    for exp in range(num_experiments):
        # 在[0,1]×[0,1]区域内均匀采样N个点
        x_samples = np.random.uniform(0, 1, N)
        y_samples = np.random.uniform(0, 1, N)
        
        # 计算函数值 f(x,y) = e^(-(x²+y²))
        function_values = np.exp(-(x_samples**2 + y_samples**2))
        
        # 蒙特卡洛二重积分估计: I ≈ (1/N) * Σ f(x_i, y_i)
        # 对于二重积分，积分区域面积为1，所以系数为1
        integral_estimate = np.mean(function_values)
        integral_estimates.append(integral_estimate)
    
    mean_integral = np.mean(integral_estimates)
    variance_integral = np.var(integral_estimates)
    
    return integral_estimates, mean_integral, variance_integral

def analytical_solution():
    """
    计算积分的解析解
    ∫₀¹ ∫₀¹ e^(-(x²+y²)) dx dy = ∫₀¹ e^(-x²) dx * ∫₀¹ e^(-y²) dy = (∫₀¹ e^(-x²) dx)²
    
    使用高斯误差函数: ∫₀¹ e^(-x²) dx = (√π/2) * erf(1)
    """
    # 使用scipy计算 ∫₀¹ e^(-x²) dx
    def integrand(x):
        return np.exp(-x**2)
    
    result, _ = integrate.quad(integrand, 0, 1)
    analytical_integral = result**2
    
    return analytical_integral

def monte_carlo_double_integration_importance_sampling(N, num_experiments=100):
    """
    使用重要性采样蒙特卡洛方法计算二重积分
    这里我们使用一个更接近函数形状的分布
    """
    integral_estimates = []
    
    for exp in range(num_experiments):
        # 使用重要性采样：采样密度与函数形状相似
        # 使用指数分布作为重要性采样分布
        u1 = np.random.uniform(0, 1, N)
        u2 = np.random.uniform(0, 1, N)
        
        # 逆变换采样得到指数分布
        x_samples = -np.log(1 - u1) / 2  # 指数分布参数为2
        y_samples = -np.log(1 - u2) / 2
        
        # 限制在[0,1]范围内
        x_samples = np.clip(x_samples, 0, 1)
        y_samples = np.clip(y_samples, 0, 1)
        
        # 计算函数值和采样密度
        function_values = np.exp(-(x_samples**2 + y_samples**2))
        sampling_density = 2 * np.exp(-2 * x_samples) * 2 * np.exp(-2 * y_samples)
        
        # 重要性采样估计
        integral_estimate = np.mean(function_values / sampling_density)
        integral_estimates.append(integral_estimate)
    
    mean_integral = np.mean(integral_estimates)
    variance_integral = np.var(integral_estimates)
    
    return integral_estimates, mean_integral, variance_integral

def visualize_double_integration(N=1000, save_path=None):
    """
    可视化二重积分蒙特卡洛过程
    
    参数:
    N: 采样点数量
    save_path: 图片保存路径
    """
    # 生成采样点
    x_samples = np.random.uniform(0, 1, N)
    y_samples = np.random.uniform(0, 1, N)
    function_values = np.exp(-(x_samples**2 + y_samples**2))
    
    # 计算积分估计值
    integral_estimate = np.mean(function_values)
    analytical_integral = analytical_solution()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 3D函数图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    x_plot = np.linspace(0, 1, 50)
    y_plot = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_plot, y_plot)
    Z = np.exp(-(X**2 + Y**2))
    
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(x_samples, y_samples, function_values, c='red', s=1, alpha=0.6)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title(f'函数 f(x,y) = e^(-(x²+y²))\n积分估计: {integral_estimate:.6f}')
    
    # 2D等高线图
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.scatter(x_samples, y_samples, c='red', s=1, alpha=0.6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('等高线图和采样点')
    ax2.set_aspect('equal')
    
    # 不同N值下的收敛情况
    ax3 = fig.add_subplot(2, 3, 3)
    N_values = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500]
    means = []
    variances = []
    
    for n in N_values:
        _, mean_integral, var_integral = monte_carlo_double_integration(n, 100)
        means.append(mean_integral)
        variances.append(var_integral)
    
    ax3.errorbar(N_values, means, yerr=np.sqrt(variances), 
                marker='o', capsize=5, capthick=2, linewidth=2, label='蒙特卡洛估计')
    ax3.axhline(y=analytical_integral, color='red', linestyle='--', linewidth=2, 
                label=f'解析解 ({analytical_integral:.6f})')
    ax3.set_xlabel('采样点数 N')
    ax3.set_ylabel('积分估计值')
    ax3.set_title('不同N值下的积分估计收敛情况')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 误差分析
    ax4 = fig.add_subplot(2, 3, 4)
    errors = [abs(mean - analytical_integral) for mean in means]
    relative_errors = [error / analytical_integral * 100 for error in errors]
    
    ax4.plot(N_values, errors, 'o-', linewidth=2, label='绝对误差')
    ax4.set_xlabel('采样点数 N')
    ax4.set_ylabel('绝对误差')
    ax4.set_title('绝对误差随N的变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    # 方差分析
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(N_values, variances, 'o-', linewidth=2, color='green', label='方差')
    ax5.set_xlabel('采样点数 N')
    ax5.set_ylabel('方差')
    ax5.set_title('方差随N的变化')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    
    # 采样点分布
    ax6 = fig.add_subplot(2, 3, 6)
    scatter = ax6.scatter(x_samples, y_samples, c=function_values, cmap='viridis', s=1, alpha=0.6)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('采样点分布（颜色表示函数值）')
    ax6.set_aspect('equal')
    plt.colorbar(scatter, ax=ax6, label='f(x,y)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    return integral_estimate

def generate_results_table():
    """
    生成不同N值下的结果表格
    """
    N_values = [10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 500]
    results = []
    
    print("正在计算不同N值下的二重积分估计值...")
    print("=" * 80)
    print(f"{'N':>4} {'Mean':>12} {'Variance':>12} {'Std_Dev':>10} {'Error':>12} {'Rel_Error(%)':>12}")
    print("-" * 80)
    
    analytical_integral = analytical_solution()
    
    for N in N_values:
        integral_estimates, mean_integral, variance_integral = monte_carlo_double_integration(N, 100)
        std_integral = np.sqrt(variance_integral)
        error = abs(mean_integral - analytical_integral)
        relative_error = error / analytical_integral * 100
        
        results.append({
            'N': N,
            'Mean_Integral': mean_integral,
            'Variance': variance_integral,
            'Std_Dev': std_integral,
            'Error': error,
            'Relative_Error(%)': relative_error
        })
        
        print(f"{N:4d} {mean_integral:12.6f} {variance_integral:12.8f} {std_integral:10.6f} {error:12.6f} {relative_error:12.4f}")
    
    print("-" * 80)
    print(f"解析解: {analytical_integral:.6f}")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv('/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q3/monte_carlo_double_integration_results.csv', 
              index=False, encoding='utf-8-sig')
    
    print("\n结果已保存到 monte_carlo_double_integration_results.csv")
    return df

def compare_sampling_methods():
    """
    比较不同采样方法的效果
    """
    N = 100
    print(f"\n比较不同采样方法 (N={N}):")
    print("=" * 60)
    
    # 均匀分布采样
    _, mean_uniform, var_uniform = monte_carlo_double_integration(N, 100)
    
    # 重要性采样
    _, mean_importance, var_importance = monte_carlo_double_integration_importance_sampling(N, 100)
    
    analytical_integral = analytical_solution()
    
    print(f"解析解: {analytical_integral:.6f}")
    print(f"均匀分布采样: 均值={mean_uniform:.6f}, 方差={var_uniform:.8f}")
    print(f"重要性采样:   均值={mean_importance:.6f}, 方差={var_importance:.8f}")
    print(f"均匀分布误差: {abs(mean_uniform - analytical_integral):.6f}")
    print(f"重要性采样误差: {abs(mean_importance - analytical_integral):.6f}")

def main():
    """
    主函数：执行完整的蒙特卡洛二重积分实验
    """
    print("蒙特卡洛方法求解二重积分 ∫₀¹ ∫₀¹ e^(-(x²+y²)) dx dy")
    print("=" * 70)
    
    # 计算解析解
    analytical_integral = analytical_solution()
    print(f"解析解: {analytical_integral:.6f}")
    print("=" * 70)
    
    # 生成结果表格
    results_df = generate_results_table()
    
    # 比较不同采样方法
    compare_sampling_methods()
    
    # 创建可视化
    print("\n正在生成可视化图片...")
    integral_estimate = visualize_double_integration(N=1000, 
                                                   save_path='/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q3/monte_carlo_double_integration_visualization.png')
    
    # 显示结果表格
    print("\n详细结果表格:")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    return results_df

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行主程序
    results = main()

