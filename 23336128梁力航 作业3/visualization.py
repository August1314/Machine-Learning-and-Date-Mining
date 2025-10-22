"""
可视化模块
用于绘制训练过程和结果的图表
"""
import matplotlib.pyplot as plt
import os

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_errors(iterations, train_errors, test_errors, title, filename):
    """
    绘制训练误差和测试误差曲线
    
    参数:
        iterations: 迭代次数列表
        train_errors: 训练误差列表
        test_errors: 测试误差列表
        title: 图表标题
        filename: 保存文件名
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_errors, 'b-', label='训练误差', linewidth=2)
    plt.plot(iterations, test_errors, 'r-', label='测试误差', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('均方误差 (MSE)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    filepath = os.path.join('analysis', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {filepath}")
    plt.close()


def plot_objective(iterations, objective_values, title, filename):
    """
    绘制目标函数值曲线
    
    参数:
        iterations: 迭代次数列表
        objective_values: 目标函数值列表
        title: 图表标题
        filename: 保存文件名
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objective_values, 'g-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('对数似然值', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    filepath = os.path.join('analysis', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {filepath}")
    plt.close()


def plot_training_size_analysis(sizes, train_errors, test_errors, filename):
    """
    绘制训练集大小分析图
    
    参数:
        sizes: 训练集大小列表
        train_errors: 训练误差列表
        test_errors: 测试误差列表
        filename: 保存文件名
    """
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_errors, 'b-', label='训练误差', linewidth=2)
    plt.plot(sizes, test_errors, 'r-', label='测试误差', linewidth=2)
    plt.xlabel('训练集大小', fontsize=12)
    plt.ylabel('误分类数量', fontsize=12)
    plt.title('训练集大小对模型性能的影响', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    filepath = os.path.join('analysis', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {filepath}")
    plt.close()
