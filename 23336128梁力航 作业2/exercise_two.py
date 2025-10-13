"""
练习二：逻辑回归实验
实现逻辑回归分类器并分析性能
"""
import numpy as np
from utils import load_data, add_intercept
from logistic_regression import LogisticRegression
from visualization import plot_objective, plot_training_size_analysis


def print_mathematical_derivation():
    """输出逻辑回归的数学推导"""
    print("\n" + "=" * 60)
    print("(a) 逻辑回归的条件对数似然公式")
    print("=" * 60)
    print("\n给定数据集有 n 个训练样本和 p 个特征")
    print("条件对数似然函数为:")
    print("\nℓ(w) = Σ[y^(i) * log(h_w(x^(i))) + (1-y^(i)) * log(1-h_w(x^(i)))]")
    print("      i=1 到 n")
    print("\n其中:")
    print("  h_w(x) = σ(w^T x) = 1 / (1 + e^(-w^T x))")
    print("  w^T x = w_0 + w_1*x_1 + w_2*x_2 + ... + w_p*x_p")
    print("  y^(i) ∈ {0, 1} 是样本 i 的真实标签")
    print("  x^(i) = [1, x_1^(i), x_2^(i), ..., x_p^(i)] 是样本 i 的特征向量")
    
    print("\n" + "=" * 60)
    print("(b) 目标函数的偏导数")
    print("=" * 60)
    print("\n对 w_0 的偏导数:")
    print("∂ℓ/∂w_0 = Σ(y^(i) - h_w(x^(i)))")
    print("         i=1 到 n")
    
    print("\n对任意 w_j (j ≥ 1) 的偏导数:")
    print("∂ℓ/∂w_j = Σ(y^(i) - h_w(x^(i))) * x_j^(i)")
    print("         i=1 到 n")
    
    print("\n参数更新规则（梯度上升）:")
    print("w_j := w_j + α * ∂ℓ/∂w_j")
    print("\n其中 α 是学习率")
    print("=" * 60)


def main():
    print("=" * 60)
    print("练习二：逻辑回归")
    print("=" * 60)
    
    # (a) 和 (b) 输出数学推导
    print_mathematical_derivation()
    
    # (c) 训练逻辑回归分类器
    print("\n" + "=" * 60)
    print("(c) 训练逻辑回归分类器")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    X_train, y_train = load_data('dataForTrainingLogistic.txt')
    X_test, y_test = load_data('dataForTestingLogistic.txt')
    
    print(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    
    # 添加截距项
    X_train_with_intercept = add_intercept(X_train)
    X_test_with_intercept = add_intercept(X_test)
    
    print(f"参数数量: {X_train_with_intercept.shape[1]} (包括截距项 w0)")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 训练模型
    print("\n开始训练...")
    model = LogisticRegression(learning_rate=0.001, n_iterations=10000)
    weights, objective_values = model.stochastic_gradient_ascent(
        X_train_with_intercept, y_train
    )
    
    print(f"\n最优参数:")
    print(f"  w0 (截距) = {weights[0]:.6f}")
    for i in range(1, len(weights)):
        print(f"  w{i} = {weights[i]:.6f}")
    
    # (d) 在测试集上评估
    print("\n" + "=" * 60)
    print("(d) 测试集评估")
    print("=" * 60)
    
    n_misclassified, accuracy = model.evaluate(X_test_with_intercept, y_test)
    print(f"\n测试集误分类样本数: {n_misclassified}")
    print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # (e) 绘制目标函数收敛曲线
    print("\n" + "=" * 60)
    print("(e) 收敛分析")
    print("=" * 60)
    
    iterations = list(range(1, len(objective_values) + 1))
    plot_objective(iterations, objective_values,
                   '逻辑回归 - 对数似然收敛曲线',
                   'logistic_convergence.png')
    
    # 判断收敛
    convergence_threshold = 0.01
    converged_iteration = None
    
    for i in range(100, len(objective_values)):
        # 检查最近100次迭代的变化
        recent_change = abs(objective_values[i] - objective_values[i-100])
        if recent_change < convergence_threshold:
            converged_iteration = i + 1
            break
    
    if converged_iteration:
        print(f"\n算法在第 {converged_iteration} 次迭代时收敛")
        print(f"(收敛标准: 最近100次迭代的对数似然变化 < {convergence_threshold})")
    else:
        print(f"\n算法在 {len(objective_values)} 次迭代内未完全收敛")
        print("建议增加迭代次数或调整学习率")
    
    print(f"\n最终对数似然值: {objective_values[-1]:.4f}")
    
    # (f) 训练集大小分析
    print("\n" + "=" * 60)
    print("(f) 训练集大小分析")
    print("=" * 60)
    
    sizes = list(range(10, 401, 10))
    train_errors_list = []
    test_errors_list = []
    
    print("\n分析不同训练集大小对性能的影响...")
    
    for k in sizes:
        # 随机选择 k 个训练样本
        np.random.seed(42)  # 固定种子以确保可重复性
        indices = np.random.choice(X_train_with_intercept.shape[0], k, replace=False)
        X_train_subset = X_train_with_intercept[indices]
        y_train_subset = y_train[indices]
        
        # 训练模型
        model_k = LogisticRegression(learning_rate=0.001, n_iterations=5000)
        model_k.weights = np.zeros(X_train_with_intercept.shape[1])
        
        # 简化训练（不打印中间结果）
        for i in range(model_k.n_iterations):
            idx = np.random.randint(0, k)
            X_i = X_train_subset[idx:idx+1]
            y_i = y_train_subset[idx:idx+1]
            prob = model_k.predict_proba(X_i)
            error = y_i - prob
            gradient = np.dot(X_i.T, error).flatten()
            model_k.weights += model_k.learning_rate * gradient
        
        # 评估
        train_error, _ = model_k.evaluate(X_train_subset, y_train_subset)
        test_error, _ = model_k.evaluate(X_test_with_intercept, y_test)
        
        train_errors_list.append(train_error)
        test_errors_list.append(test_error)
        
        if k % 50 == 0:
            print(f"训练集大小 {k}: 训练误差 = {train_error}, 测试误差 = {test_error}")
    
    # 绘制训练集大小分析图
    plot_training_size_analysis(sizes, train_errors_list, test_errors_list,
                                'logistic_training_size_analysis.png')
    
    print("\n分析结果:")
    print("- 随着训练集大小增加，训练误差通常会上升")
    print("  (因为模型更难完美拟合更多样本)")
    print("- 随着训练集大小增加，测试误差通常会下降")
    print("  (因为模型学到了更多信息，泛化能力更强)")
    print("- 当训练集足够大时，两条曲线会趋于稳定")
    print("  (表明模型已经充分学习)")
    
    print("\n" + "=" * 60)
    print("练习二完成！所有图表已保存到 analysis/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()
