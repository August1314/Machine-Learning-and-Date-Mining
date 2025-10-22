"""
SVM分类器实现 - 作业4
使用随机梯度下降(SGD)从零实现支持向量机

SVM关键概念:
1. 目标: 找到最大间隔超平面来分离两类数据
2. 决策函数: f(x) = sign(w^T * x + b)
3. 损失函数: Hinge Loss + L2正则化
4. 优化方法: 随机梯度下降(SGD)

实现方法概述:
- 使用NumPy进行数值计算
- 不使用sklearn等预构建的SVM实现
- 按照作业提供的梯度公式进行参数更新
- 支持不同的正则化常数λ进行实验
"""

import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_data(filepath):
    """
    从文本文件加载数据
    
    参数:
        filepath: 数据文件路径
    
    返回:
        features: 原始特征列表
        labels: 标签列表（测试集返回None）
    """
    features = []
    labels = []
    has_labels = 'train' in filepath.lower() or 'ground_truth' in filepath.lower()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除首尾空格并按逗号分割
            parts = [p.strip() for p in line.strip().split(',')]
            
            # 过滤掉空字符串
            parts = [p for p in parts if p]
            
            if has_labels:
                # 训练集或ground_truth: 最后一列是标签
                features.append(parts[:-1])
                labels.append(parts[-1])
            else:
                # 测试集: 没有标签
                features.append(parts)
    
    return features, labels if has_labels else None


print("=" * 60)
print("SVM分类器 - Adult Income数据集分类")
print("=" * 60)
print()

# 加载数据
print("正在加载数据...")
train_features, train_labels = load_data('train.txt')
test_features, test_labels = load_data('test_ground_truth.txt')

print(f"训练集样本数: {len(train_features)}")
print(f"测试集样本数: {len(test_features)}")
print(f"特征数量: {len(train_features[0])}")
print()

class DataPreprocessor:
    """
    数据预处理器
    负责特征编码、标准化和标签转换
    """
    
    def __init__(self):
        """初始化预处理器"""
        # 存储训练集的统计信息
        self.feature_means = None  # 数值特征的均值
        self.feature_stds = None   # 数值特征的标准差
        self.categorical_mappings = {}  # 分类特征的映射字典
        self.numerical_indices = []  # 数值特征的索引
        self.categorical_indices = []  # 分类特征的索引
        
    def _identify_feature_types(self, features_sample):
        """
        识别特征类型（数值或分类）
        
        参数:
            features_sample: 样本特征列表
        """
        # Adult数据集的特征索引
        # 0: age (数值)
        # 1: workclass (分类)
        # 2: fnlwgt (数值)
        # 3: education (分类)
        # 4: education-num (数值)
        # 5: marital-status (分类)
        # 6: occupation (分类)
        # 7: relationship (分类)
        # 8: race (分类)
        # 9: sex (分类)
        # 10: capital-gain (数值)
        # 11: capital-loss (数值)
        # 12: hours-per-week (数值)
        # 13: native-country (分类)
        
        self.numerical_indices = [0, 2, 4, 10, 11, 12]
        self.categorical_indices = [1, 3, 5, 6, 7, 8, 9, 13]

    def _encode_categorical_features(self, features, is_training=True):
        """
        编码分类特征为数值
        
        参数:
            features: 原始特征列表
            is_training: 是否是训练阶段
        
        返回:
            编码后的特征列表
        """
        encoded_features = []
        
        for row in features:
            encoded_row = row.copy()
            
            for idx in self.categorical_indices:
                value = row[idx]
                
                # 处理缺失值
                if value == '?':
                    value = 'MISSING'
                
                if is_training:
                    # 训练阶段: 建立映射
                    if idx not in self.categorical_mappings:
                        self.categorical_mappings[idx] = {}
                    
                    if value not in self.categorical_mappings[idx]:
                        # 为新类别分配编码
                        self.categorical_mappings[idx][value] = len(self.categorical_mappings[idx])
                    
                    encoded_row[idx] = self.categorical_mappings[idx][value]
                else:
                    # 测试阶段: 使用已有映射
                    if idx in self.categorical_mappings:
                        if value in self.categorical_mappings[idx]:
                            encoded_row[idx] = self.categorical_mappings[idx][value]
                        else:
                            # 未见过的类别，使用特殊编码
                            encoded_row[idx] = -1
                    else:
                        encoded_row[idx] = -1
            
            encoded_features.append(encoded_row)
        
        return encoded_features

    def _convert_to_numerical(self, features):
        """
        将所有特征转换为浮点数
        
        参数:
            features: 特征列表
        
        返回:
            numpy数组
        """
        numerical_features = []
        
        for row in features:
            numerical_row = []
            for val in row:
                try:
                    numerical_row.append(float(val))
                except:
                    numerical_row.append(0.0)  # 无法转换的值设为0
            numerical_features.append(numerical_row)
        
        return np.array(numerical_features)
    
    def _standardize_features(self, X, is_training=True):
        """
        标准化数值特征 (Z-score标准化)
        
        参数:
            X: 特征矩阵
            is_training: 是否是训练阶段
        
        返回:
            标准化后的特征矩阵
        """
        X_standardized = X.copy()
        
        if is_training:
            # 训练阶段: 计算均值和标准差
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            # 避免除以零
            self.feature_stds[self.feature_stds == 0] = 1.0
        
        # 标准化: (x - μ) / σ
        X_standardized = (X_standardized - self.feature_means) / self.feature_stds
        
        return X_standardized

    def _encode_labels(self, labels):
        """
        编码标签: "<=50K" -> -1, ">50K" -> +1
        
        参数:
            labels: 原始标签列表
        
        返回:
            编码后的标签数组
        """
        encoded_labels = []
        for label in labels:
            if '<=50K' in label or '<50K' in label:
                encoded_labels.append(-1)
            else:  # ">50K" or ">=50K"
                encoded_labels.append(1)
        
        return np.array(encoded_labels)
    
    def fit_transform(self, features, labels):
        """
        拟合并转换训练数据
        
        参数:
            features: 原始特征列表
            labels: 原始标签列表
        
        返回:
            X: 标准化后的特征矩阵
            y: 编码后的标签向量
        """
        # 识别特征类型
        self._identify_feature_types(features[0])
        
        # 编码分类特征
        encoded_features = self._encode_categorical_features(features, is_training=True)
        
        # 转换为数值矩阵
        X = self._convert_to_numerical(encoded_features)
        
        # 标准化特征
        X = self._standardize_features(X, is_training=True)
        
        # 编码标签
        y = self._encode_labels(labels)
        
        return X, y
    
    def transform(self, features):
        """
        转换测试数据（使用训练集的统计信息）
        
        参数:
            features: 原始特征列表
        
        返回:
            X: 标准化后的特征矩阵
        """
        # 编码分类特征
        encoded_features = self._encode_categorical_features(features, is_training=False)
        
        # 转换为数值矩阵
        X = self._convert_to_numerical(encoded_features)
        
        # 标准化特征
        X = self._standardize_features(X, is_training=False)
        
        return X


class SVMClassifier:
    """
    支持向量机分类器
    使用随机梯度下降(SGD)训练
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_epochs=100):
        """
        初始化SVM分类器
        
        参数:
            learning_rate: 学习率η
            lambda_param: 正则化常数λ
            n_epochs: 训练轮数
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.weights = None  # 权重向量a
        self.bias = None     # 偏置项b

    def fit(self, X, y):
        """
        使用SGD训练SVM
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,) 值为-1或+1
        """
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # SGD训练
        for epoch in range(self.n_epochs):
            # 随机打乱样本顺序
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                x_i = X[idx]
                y_i = y[idx]
                
                # 计算间隔: y_i * (w^T * x_i + b)
                margin = y_i * (np.dot(self.weights, x_i) + self.bias)
                
                # 根据作业提供的梯度公式更新参数
                if margin >= 1:
                    # 样本在边界外，只更新正则化项
                    grad_w = self.lambda_param * self.weights
                    grad_b = 0
                else:
                    # 样本违反边界约束
                    grad_w = self.lambda_param * self.weights - y_i * x_i
                    grad_b = -y_i
                
                # 更新参数
                self.weights = self.weights - self.learning_rate * grad_w
                self.bias = self.bias - self.learning_rate * grad_b
            
            # 每10个epoch打印一次进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # 计算训练集准确率
                train_predictions = self.predict(X)
                train_accuracy = np.mean(train_predictions == y) * 100
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, 训练准确率: {train_accuracy:.2f}%")

    def predict(self, X):
        """
        预测新样本
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
        
        返回:
            预测标签 (n_samples,) 值为-1或+1
        """
        # 计算决策函数: w^T * x + b
        decision_scores = np.dot(X, self.weights) + self.bias
        
        # 应用sign函数
        predictions = np.where(decision_scores >= 0, 1, -1)
        
        return predictions

    def score(self, X, y):
        """
        计算准确率
        
        参数:
            X: 特征矩阵
            y: 真实标签
        
        返回:
            准确率 (0-100之间的百分比)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
        model: 训练好的SVM模型
        X_test: 测试特征矩阵
        y_test: 测试标签（如果有的话）
    
    返回:
        包含评估指标的字典
    """
    predictions = model.predict(X_test)
    
    if y_test is not None:
        accuracy = model.score(X_test, y_test)
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }
    else:
        return {
            'predictions': predictions
        }


def regularization_analysis(X_train, y_train, X_test, y_test, lambda_values, learning_rate=0.001, n_epochs=100):
    """
    对不同λ值进行实验分析
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签
        X_test: 测试特征矩阵
        y_test: 测试标签
        lambda_values: 要测试的λ值列表
        learning_rate: 学习率
        n_epochs: 训练轮数
    
    返回:
        {lambda: accuracy} 字典
    """
    results = {}
    
    print("\n" + "=" * 60)
    print("正则化分析: 测试不同的λ值")
    print("=" * 60)
    
    for lambda_val in lambda_values:
        print(f"\n训练 SVM (λ = {lambda_val})...")
        
        # 创建并训练模型
        model = SVMClassifier(
            learning_rate=learning_rate,
            lambda_param=lambda_val,
            n_epochs=n_epochs
        )
        model.fit(X_train, y_train)
        
        # 评估模型
        if y_test is not None:
            accuracy = model.score(X_test, y_test)
            results[lambda_val] = accuracy
            print(f"测试准确率: {accuracy:.2f}%")
        else:
            print("测试集无标签，无法计算准确率")
    
    return results


def generate_analysis_explanation(results):
    """
    生成正则化分析的解释说明
    
    参数:
        results: {lambda: accuracy} 字典
    
    返回:
        解释文本
    """
    explanation = "\n正则化分析说明:\n"
    explanation += "=" * 60 + "\n\n"
    
    # 排序结果
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    explanation += "不同λ值的测试准确率:\n"
    for lambda_val, accuracy in sorted_results:
        explanation += f"  λ = {lambda_val:6.3f}: 准确率 = {accuracy:.2f}%\n"
    
    explanation += "\n分析:\n"
    explanation += "正则化常数λ控制模型复杂度和泛化能力之间的平衡。\n\n"
    
    explanation += "1. λ较小(如0.001):\n"
    explanation += "   - 正则化惩罚较弱，模型更关注最小化训练误差\n"
    explanation += "   - 可能导致过拟合，在训练集上表现好但测试集上可能较差\n"
    explanation += "   - 决策边界可能过于复杂\n\n"
    
    explanation += "2. λ适中(如0.01-0.1):\n"
    explanation += "   - 在训练误差和模型复杂度之间取得平衡\n"
    explanation += "   - 通常能获得较好的泛化性能\n"
    explanation += "   - 这个范围内的λ值往往表现最佳\n\n"
    
    explanation += "3. λ较大(如1.0):\n"
    explanation += "   - 正则化惩罚较强，模型更简单\n"
    explanation += "   - 可能导致欠拟合，模型过于简单无法捕捉数据模式\n"
    explanation += "   - 训练和测试准确率都可能下降\n\n"
    
    # 找出最佳λ值
    best_lambda = max(sorted_results, key=lambda x: x[1])
    explanation += f"在本次实验中，λ = {best_lambda[0]} 获得了最高的测试准确率 {best_lambda[1]:.2f}%\n"
    
    return explanation


def plot_regularization_results(results, output_path='regularization_analysis.png'):
    """
    绘制正则化分析结果图表
    
    参数:
        results: {lambda: accuracy} 字典
        output_path: 输出图片路径
    """
    # 排序结果
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    lambda_values = [x[0] for x in sorted_results]
    accuracies = [x[1] for x in sorted_results]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 绘制折线图
    plt.plot(lambda_values, accuracies, 'b-o', linewidth=2, markersize=8, label='测试准确率')
    
    # 标记每个点的准确率值
    for i, (lambda_val, acc) in enumerate(zip(lambda_values, accuracies)):
        plt.annotate(f'{acc:.2f}%', 
                    xy=(lambda_val, acc), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 设置对数刻度（因为λ值跨度大）
    plt.xscale('log')
    
    # 设置标签和标题
    plt.xlabel('正则化常数 λ (对数刻度)', fontsize=12)
    plt.ylabel('测试准确率 (%)', fontsize=12)
    plt.title('SVM正则化分析: λ值对测试准确率的影响', fontsize=14, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(fontsize=11)
    
    # 设置y轴范围，留出一些空间
    y_min = min(accuracies) - 2
    y_max = max(accuracies) + 2
    plt.ylim(y_min, y_max)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    # 关闭图表以释放内存
    plt.close()


def save_results_summary(results, X_train, X_test, output_path='results_summary.txt'):
    """
    保存结果摘要到文件
    
    参数:
        results: 正则化分析结果字典
        X_train: 训练特征矩阵
        X_test: 测试特征矩阵
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("SVM分类器训练结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("数据集信息:\n")
        f.write(f"- 训练样本数: {X_train.shape[0]}\n")
        f.write(f"- 测试样本数: {X_test.shape[0]}\n")
        f.write(f"- 特征维度: {X_train.shape[1]}\n\n")
        
        f.write("正则化分析结果:\n")
        f.write("-" * 60 + "\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[0])
        for lambda_val, accuracy in sorted_results:
            f.write(f"λ = {lambda_val:6.3f}: 准确率 = {accuracy:.2f}%\n")
        
        f.write("\n")
        explanation = generate_analysis_explanation(results)
        f.write(explanation)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    """
    主程序入口
    """
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    print("步骤1: 数据预处理")
    print("-" * 60)
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 预处理训练数据
    print("正在预处理训练数据...")
    X_train, y_train = preprocessor.fit_transform(train_features, train_labels)
    print(f"训练集形状: {X_train.shape}")
    print(f"标签分布: {np.sum(y_train == 1)} 个 >50K, {np.sum(y_train == -1)} 个 <=50K")
    
    # 预处理测试数据（使用训练集的统计信息）
    print("\n正在预处理测试数据...")
    X_test = preprocessor.transform(test_features)
    y_test = preprocessor._encode_labels(test_labels)
    print(f"测试集形状: {X_test.shape}")
    print(f"测试集标签分布: {np.sum(y_test == 1)} 个 >50K, {np.sum(y_test == -1)} 个 <=50K")
    
    print("\n使用完整训练集进行训练")
    
    # 步骤2: 正则化分析
    print("\n" + "=" * 60)
    print("步骤2: 正则化分析")
    print("=" * 60)
    
    lambda_values = [1e-3, 1e-2, 1e-1, 1]
    results = regularization_analysis(
        X_train, y_train,
        X_test, y_test,
        lambda_values,
        learning_rate=0.001,
        n_epochs=50  # 使用较少的epochs以加快速度
    )
    
    # 步骤3: 显示结果
    print("\n" + "=" * 60)
    print("步骤3: 结果总结")
    print("=" * 60)
    
    print("\n正则化分析结果:")
    print("-" * 60)
    for lambda_val, accuracy in sorted(results.items()):
        print(f"λ = {lambda_val:6.3f}: 测试准确率 = {accuracy:.2f}%")
    
    # 生成并打印解释
    explanation = generate_analysis_explanation(results)
    print(explanation)
    
    # 步骤4: 保存结果
    print("\n" + "=" * 60)
    print("步骤4: 保存结果")
    print("=" * 60)
    
    save_results_summary(results, X_train, X_test)
    
    # 生成可视化图表
    print("\n生成正则化分析图表...")
    plot_regularization_results(results)
    
    # 步骤5: 使用最佳λ值在完整训练集上训练最终模型
    print("\n" + "=" * 60)
    print("步骤5: 训练最终模型")
    print("=" * 60)
    
    best_lambda = max(results.items(), key=lambda x: x[1])[0]
    print(f"\n使用最佳λ值 {best_lambda} 在完整训练集上训练最终模型...")
    
    final_model = SVMClassifier(
        learning_rate=0.001,
        lambda_param=best_lambda,
        n_epochs=50
    )
    final_model.fit(X_train, y_train)
    
    # 在测试集上进行预测和评估
    print("\n在测试集上进行预测和评估...")
    test_predictions = final_model.predict(X_test)
    final_accuracy = final_model.score(X_test, y_test)
    print(f"预测完成: {len(test_predictions)} 个样本")
    print(f"预测为 >50K 的样本数: {np.sum(test_predictions == 1)}")
    print(f"预测为 <=50K 的样本数: {np.sum(test_predictions == -1)}")
    print(f"\n最终测试准确率: {final_accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("程序执行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
