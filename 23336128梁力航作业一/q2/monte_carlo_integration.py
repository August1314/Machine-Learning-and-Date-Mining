import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def monte_carlo_integration(N, num_experiments=100):
    """
    使用蒙特卡洛方法计算积分 ∫₀¹ e^x dx
    
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
        # 方法1: 均匀分布采样 (最简单的方法)
        # 在[0,1]区间内均匀采样N个点
        x_samples = np.random.uniform(0, 1, N)
        
        # 计算函数值 f(x) = e^x
        function_values = np.exp(x_samples)
        
        # 蒙特卡洛积分估计: I ≈ (1/N) * Σ f(x_i)
        integral_estimate = np.mean(function_values)
        integral_estimates.append(integral_estimate)
    
    mean_integral = np.mean(integral_estimates)
    variance_integral = np.var(integral_estimates)
    
    return integral_estimates, mean_integral, variance_integral

def monte_carlo_integration_importance_sampling(N, num_experiments=100):
    """
    使用重要性采样蒙特卡洛方法计算积分 ∫₀¹ e^x dx
    
    这里我们使用一个更接近e^x形状的分布作为重要性采样分布
    例如: p(x) = (e-1) * e^x / (e-1) = e^x / (e-1) 在[0,1]上的归一化分布
    但实际上我们使用一个简单的线性分布作为示例
    """
    integral_estimates = []
    
    for exp in range(num_experiments):
        # 使用重要性采样：采样密度与函数形状相似
        # 这里使用一个简单的线性分布 p(x) = 2x (在[0,1]上归一化)
        u = np.random.uniform(0, 1, N)
        x_samples = np.sqrt(u)  # 逆变换采样得到 p(x) = 2x 的分布
        
        # 计算函数值和采样密度
        function_values = np.exp(x_samples)
        sampling_density = 2 * x_samples  # p(x) = 2x
        
        # 重要性采样估计: I ≈ (1/N) * Σ [f(x_i) / p(x_i)]
        integral_estimate = np.mean(function_values / sampling_density)
        integral_estimates.append(integral_estimate)
    
    mean_integral = np.mean(integral_estimates)
    variance_integral = np.var(integral_estimates)
    
    return integral_estimates, mean_integral, variance_integral

def visualize_integration(N=1000, save_path=None):
    """
    可视化蒙特卡洛积分过程
    
    参数:
    N: 采样点数量
    save_path: 图片保存路径
    """
    # 生成采样点
    x_samples = np.random.uniform(0, 1, N)
    function_values = np.exp(x_samples)
    
    # 计算积分估计值
    integral_estimate = np.mean(function_values)
    true_integral = np.e - 1
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 左上图：函数曲线和采样点
    x_plot = np.linspace(0, 1, 1000)
    y_plot = np.exp(x_plot)
    
    ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = e^x')
    ax1.scatter(x_samples, function_values, c='red', s=1, alpha=0.6, label=f'采样点 (N={N})')
    ax1.fill_between(x_plot, y_plot, alpha=0.3, color='blue', label='积分区域')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(f'蒙特卡洛积分: ∫₀¹ e^x dx\n估计值: {integral_estimate:.6f}, 真实值: {true_integral:.6f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右上图：不同N值下的收敛情况
    N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    means = []
    variances = []
    
    for n in N_values:
        _, mean_integral, var_integral = monte_carlo_integration(n, 100)
        means.append(mean_integral)
        variances.append(var_integral)
    
    ax2.errorbar(N_values, means, yerr=np.sqrt(variances), 
                marker='o', capsize=5, capthick=2, linewidth=2, label='蒙特卡洛估计')
    ax2.axhline(y=true_integral, color='red', linestyle='--', linewidth=2, 
                label=f'真实值 ({true_integral:.6f})')
    ax2.set_xlabel('采样点数 N')
    ax2.set_ylabel('积分估计值')
    ax2.set_title('不同N值下的积分估计收敛情况')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 左下图：误差分析
    errors = [abs(mean - true_integral) for mean in means]
    relative_errors = [error / true_integral * 100 for error in errors]
    
    ax3.plot(N_values, errors, 'o-', linewidth=2, label='绝对误差')
    ax3.set_xlabel('采样点数 N')
    ax3.set_ylabel('绝对误差')
    ax3.set_title('绝对误差随N的变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 右下图：方差分析
    ax4.plot(N_values, variances, 'o-', linewidth=2, color='green', label='方差')
    ax4.set_xlabel('采样点数 N')
    ax4.set_ylabel('方差')
    ax4.set_title('方差随N的变化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
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
    N_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    results = []
    
    print("正在计算不同N值下的积分估计值...")
    print("=" * 70)
    print(f"{'N':>4} {'Mean':>10} {'Variance':>12} {'Std_Dev':>10} {'Error':>10} {'Rel_Error(%)':>12}")
    print("-" * 70)
    
    true_integral = np.e - 1
    
    for N in N_values:
        integral_estimates, mean_integral, variance_integral = monte_carlo_integration(N, 100)
        std_integral = np.sqrt(variance_integral)
        error = abs(mean_integral - true_integral)
        relative_error = error / true_integral * 100
        
        results.append({
            'N': N,
            'Mean_Integral': mean_integral,
            'Variance': variance_integral,
            'Std_Dev': std_integral,
            'Error': error,
            'Relative_Error(%)': relative_error
        })
        
        print(f"{N:4d} {mean_integral:10.6f} {variance_integral:12.8f} {std_integral:10.6f} {error:10.6f} {relative_error:12.4f}")
    
    print("-" * 70)
    print(f"真实积分值: {true_integral:.6f}")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv('/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q2/monte_carlo_integration_results.csv', 
              index=False, encoding='utf-8-sig')
    
    print("\n结果已保存到 monte_carlo_integration_results.csv")
    return df

def compare_sampling_methods():
    """
    比较不同采样方法的效果
    """
    N = 100
    print(f"\n比较不同采样方法 (N={N}):")
    print("=" * 50)
    
    # 均匀分布采样
    _, mean_uniform, var_uniform = monte_carlo_integration(N, 100)
    
    # 重要性采样
    _, mean_importance, var_importance = monte_carlo_integration_importance_sampling(N, 100)
    
    true_integral = np.e - 1
    
    print(f"真实积分值: {true_integral:.6f}")
    print(f"均匀分布采样: 均值={mean_uniform:.6f}, 方差={var_uniform:.8f}")
    print(f"重要性采样:   均值={mean_importance:.6f}, 方差={var_importance:.8f}")
    print(f"均匀分布误差: {abs(mean_uniform - true_integral):.6f}")
    print(f"重要性采样误差: {abs(mean_importance - true_integral):.6f}")

def main():
    """
    主函数：执行完整的蒙特卡洛积分实验
    """
    print("蒙特卡洛方法求解积分 ∫₀¹ e^x dx")
    print("=" * 50)
    print(f"真实积分值: e - 1 = {np.e - 1:.6f}")
    print("=" * 50)
    
    # 生成结果表格
    results_df = generate_results_table()
    
    # 比较不同采样方法
    compare_sampling_methods()
    
    # 创建可视化
    print("\n正在生成可视化图片...")
    integral_estimate = visualize_integration(N=1000, 
                                            save_path='/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q2/monte_carlo_integration_visualization.png')
    
    # 显示结果表格
    print("\n详细结果表格:")
    print(results_df.to_string(index=False, float_format='%.6f'))
    
    return results_df

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行主程序
    results = main()

