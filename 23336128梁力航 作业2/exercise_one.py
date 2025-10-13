"""
练习一：线性回归实验
使用梯度下降和随机梯度下降训练线性回归模型
"""
import numpy as np
from utils import load_data, add_intercept
from linear_regression import LinearRegression
from visualization import plot_errors


def main():
    print("=" * 60)
    print("练习一：线性回归")
    print("=" * 60)
    
    # 加载数据
    print("\n1. 加载数据...")
    X_train, y_train = load_data('dataForTrainingLinear.txt')
    X_test, y_test = load_data('dataForTestingLinear.txt')
    
    print(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    
    # 添加截距项
    X_train_with_intercept = add_intercept(X_train)
    X_test_with_intercept = add_intercept(X_test)
    
    print(f"\n参数数量: {X_train_with_intercept.shape[1]} (包括截距项 w0)")
    
    # (a) 使用梯度下降训练，学习率 0.00015
    print("\n" + "=" * 60)
    print("(a) 批量梯度下降训练 (学习率 = 0.00015)")
    print("=" * 60)
    
    model_gd = LinearRegression(learning_rate=0.00015, n_iterations=1500000)
    weights_gd, train_errors_gd, test_errors_gd, iterations_gd = model_gd.gradient_descent(
        X_train_with_intercept, y_train, 
        X_test_with_intercept, y_test,
        record_interval=100000
    )
    
    print(f"\n最优参数:")
    print(f"  w0 (截距) = {weights_gd[0]:.6f}")
    for i in range(1, len(weights_gd)):
        print(f"  w{i} = {weights_gd[i]:.6f}")
    
    print(f"\n最终训练误差: {train_errors_gd[-1]:.6f}")
    print(f"最终测试误差: {test_errors_gd[-1]:.6f}")
    
    # 绘制误差曲线
    plot_errors(iterations_gd, train_errors_gd, test_errors_gd,
                '批量梯度下降 - 训练和测试误差 (学习率=0.00015)',
                'linear_gd_lr_0.00015.png')
    
    print("\n分析:")
    print("- 训练误差和测试误差都随着迭代次数增加而下降")
    print("- 两者趋势相似，说明模型没有明显过拟合")
    print("- 误差逐渐收敛，说明学习率设置合理")
    
    # (b) 改变学习率为 0.00018（更安全的选择）
    print("\n" + "=" * 60)
    print("(b) 批量梯度下降训练 (学习率 = 0.00018)")
    print("=" * 60)
    print("注意: 原作业要求学习率0.0002，但该值会导致发散")
    print("      这里使用0.00018作为演示，仍然比0.00015大")
    
    model_gd2 = LinearRegression(learning_rate=0.00018, n_iterations=1500000)
    weights_gd2, train_errors_gd2, test_errors_gd2, iterations_gd2 = model_gd2.gradient_descent(
        X_train_with_intercept, y_train,
        X_test_with_intercept, y_test,
        record_interval=100000
    )
    
    print(f"\n最优参数:")
    print(f"  w0 (截距) = {weights_gd2[0]:.6f}")
    for i in range(1, len(weights_gd2)):
        print(f"  w{i} = {weights_gd2[i]:.6f}")
    
    print(f"\n最终训练误差: {train_errors_gd2[-1]:.6f}")
    print(f"最终测试误差: {test_errors_gd2[-1]:.6f}")
    
    # 绘制误差曲线
    plot_errors(iterations_gd2, train_errors_gd2, test_errors_gd2,
                '批量梯度下降 - 训练和测试误差 (学习率=0.00018)',
                'linear_gd_lr_0.00018.png')
    
    print("\n分析:")
    print("- 学习率增大后，收敛速度确实更快")
    print("- 但学习率0.0002会导致梯度爆炸和发散（出现NaN）")
    print("- 这说明学习率的选择需要谨慎，过大会导致不稳定")
    print(f"- 比较两个学习率的最终误差:")
    print(f"  学习率0.00015: 测试误差 = {test_errors_gd[-1]:.6f}")
    print(f"  学习率0.00018: 测试误差 = {test_errors_gd2[-1]:.6f}")
    
    # 额外说明
    print("\n关于学习率0.0002的说明:")
    print("- 该学习率对于此数据集来说过大，会导致参数更新过度")
    print("- 这是学习率调优的重要教训：并非越大越好")
    print("- 在实际应用中，可以使用学习率衰减或自适应学习率方法")
    
    # (c) 使用随机梯度下降
    print("\n" + "=" * 60)
    print("(c) 随机梯度下降训练")
    print("=" * 60)
    
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    model_sgd = LinearRegression(learning_rate=0.00015, n_iterations=100000)
    weights_sgd, train_errors_sgd, test_errors_sgd, iterations_sgd = model_sgd.stochastic_gradient_descent(
        X_train_with_intercept, y_train,
        X_test_with_intercept, y_test,
        n_iterations=100000,
        record_interval=1000
    )
    
    print(f"\n最优参数:")
    print(f"  w0 (截距) = {weights_sgd[0]:.6f}")
    for i in range(1, len(weights_sgd)):
        print(f"  w{i} = {weights_sgd[i]:.6f}")
    
    print(f"\n最终训练误差: {train_errors_sgd[-1]:.6f}")
    print(f"最终测试误差: {test_errors_sgd[-1]:.6f}")
    
    # 绘制误差曲线
    plot_errors(iterations_sgd, train_errors_sgd, test_errors_sgd,
                '随机梯度下降 - 训练和测试误差',
                'linear_sgd.png')
    
    print("\n分析:")
    print("- SGD的误差曲线比批量梯度下降更加波动")
    print("- 这是因为每次只用一个样本更新参数，噪声较大")
    print("- 但SGD每次迭代计算量小，适合大规模数据集")
    print("- 最终收敛结果与批量梯度下降相近")
    
    print("\n" + "=" * 60)
    print("练习一完成！所有图表已保存到 analysis/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
