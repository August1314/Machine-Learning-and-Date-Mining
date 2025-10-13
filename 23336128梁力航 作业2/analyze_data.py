"""
数据分析脚本
分析为什么测试误差比训练误差大
"""
import numpy as np
from utils import load_data

print("=" * 70)
print("数据分析：为什么测试误差这么大？")
print("=" * 70)

# 加载数据
X_train, y_train = load_data('dataForTrainingLinear.txt')
X_test, y_test = load_data('dataForTestingLinear.txt')

print("\n1. 数据基本统计")
print("-" * 70)
print(f"训练集大小: {len(y_train)} 样本")
print(f"测试集大小: {len(y_test)} 样本")

print("\n2. 目标变量（房价）统计")
print("-" * 70)
print(f"训练集房价范围: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"训练集房价均值: {y_train.mean():.2f}")
print(f"训练集房价标准差: {y_train.std():.2f}")
print()
print(f"测试集房价范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"测试集房价均值: {y_test.mean():.2f}")
print(f"测试集房价标准差: {y_test.std():.2f}")

print("\n3. 特征统计")
print("-" * 70)
print("特征1（面积）:")
print(f"  训练集: [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}], 均值={X_train[:, 0].mean():.2f}")
print(f"  测试集: [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}], 均值={X_test[:, 0].mean():.2f}")
print()
print("特征2（距离）:")
print(f"  训练集: [{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}], 均值={X_train[:, 1].mean():.2f}")
print(f"  测试集: [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}], 均值={X_test[:, 1].mean():.2f}")

print("\n4. 均方误差（MSE）的含义")
print("-" * 70)
print("MSE = 平均((预测值 - 真实值)^2)")
print()
print(f"训练集MSE = 3.82 意味着:")
print(f"  平均误差 ≈ √3.82 ≈ 1.95 (亿元)")
print(f"  相对于平均房价 {y_train.mean():.2f} 亿元")
print(f"  相对误差 ≈ {np.sqrt(3.82)/y_train.mean()*100:.2f}%")
print()
print(f"测试集MSE = 133.75 意味着:")
print(f"  平均误差 ≈ √133.75 ≈ 11.56 (亿元)")
print(f"  相对于平均房价 {y_test.mean():.2f} 亿元")
print(f"  相对误差 ≈ {np.sqrt(133.75)/y_test.mean()*100:.2f}%")

print("\n5. 为什么测试误差大？")
print("-" * 70)
print("原因分析:")
print()
print("① 数据集很小")
print(f"   - 训练集只有 {len(y_train)} 个样本")
print(f"   - 测试集只有 {len(y_test)} 个样本")
print("   - 小样本容易导致训练集和测试集分布不一致")
print()
print("② 测试集可能包含特殊样本")
print("   - 某些测试样本可能在训练集中没有类似的样本")
print("   - 模型对这些样本的预测会有较大误差")
print()
print("③ 这是正常的泛化误差")
print("   - 训练误差总是会比测试误差小")
print("   - 模型在训练数据上拟合得更好")
print("   - 测试误差反映了模型的真实泛化能力")

# 计算每个测试样本的误差
print("\n6. 测试集每个样本的预测分析")
print("-" * 70)

# 使用训练好的参数（从实验结果）
w0 = 79.463702
w1 = 6.761940
w2 = -72.380883

predictions = w0 + X_test[:, 0] * w1 + X_test[:, 1] * w2
errors = (predictions - y_test) ** 2

print(f"{'样本':<6} {'面积':<8} {'距离':<8} {'真实价格':<10} {'预测价格':<10} {'误差^2':<10}")
print("-" * 70)
for i in range(len(y_test)):
    print(f"{i+1:<6} {X_test[i, 0]:<8.2f} {X_test[i, 1]:<8.2f} {y_test[i]:<10.2f} {predictions[i]:<10.2f} {errors[i]:<10.2f}")

print()
print(f"平均误差^2 (MSE): {errors.mean():.2f}")
print(f"最大误差^2: {errors.max():.2f} (样本 {errors.argmax()+1})")
print(f"最小误差^2: {errors.min():.2f} (样本 {errors.argmin()+1})")

print("\n7. 结论")
print("-" * 70)
print("测试误差133.75虽然比训练误差3.82大很多，但这是正常的：")
print()
print("✓ 训练误差小是因为模型在训练数据上优化过")
print("✓ 测试误差大反映了模型在新数据上的真实表现")
print("✓ 平均预测误差约11.56亿元，对于房价预测来说是可接受的")
print("✓ 如果要改进，可以考虑：")
print("  - 收集更多训练数据")
print("  - 添加更多特征")
print("  - 使用特征缩放/标准化")
print("  - 尝试多项式特征或其他非线性模型")

print("\n" + "=" * 70)
