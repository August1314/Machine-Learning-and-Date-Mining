# 设计文档

## 概述

本设计文档描述了从零实现SVM分类器的技术方案。系统将使用随机梯度下降（SGD）训练SVM模型，对Adult Income数据集进行二分类。整个实现将使用Python和NumPy，不依赖sklearn等预构建的机器学习库。

## 架构

### 系统组件

系统采用模块化设计，包含以下主要组件：

```
Homework 4/
├── svm_classifier.py          # 主程序文件
├── train.txt                  # 训练数据
├── test.txt                   # 测试数据
├── results_summary.txt        # 结果摘要文件
└── regularization_analysis.png # 可视化图表（可选）
```

### 数据流

1. **数据加载** → 2. **数据预处理** → 3. **SVM训练** → 4. **模型评估** → 5. **结果输出**

## 组件和接口

### 1. 数据加载模块

**功能**: 从文本文件加载数据

**接口**:
```python
def load_data(filepath: str) -> tuple:
    """
    加载数据文件
    
    参数:
        filepath: 数据文件路径
    
    返回:
        (features, labels) 或 (features, None) 对于测试集
    """
    pass
```

**实现细节**:
- 使用Python内置的文件读取功能
- 按逗号分隔每行数据
- 区分训练集（有标签）和测试集（无标签）
- 处理数据中的空格

### 2. 数据预处理模块

**功能**: 特征工程和数据标准化

**接口**:
```python
class DataPreprocessor:
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.categorical_mappings = {}
    
    def fit(self, X_raw: list) -> None:
        """学习训练集的统计信息"""
        pass
    
    def transform(self, X_raw: list) -> np.ndarray:
        """转换数据为数值特征矩阵"""
        pass
    
    def fit_transform(self, X_raw: list) -> np.ndarray:
        """拟合并转换训练数据"""
        pass
```

**实现细节**:
- **特征类型识别**: 
  - 数值特征: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
  - 分类特征: workclass, education, marital-status, occupation, relationship, race, sex, native-country
  
- **分类特征编码**:
  - 使用标签编码（Label Encoding）将分类值映射为整数
  - 为每个分类特征维护一个映射字典
  - 处理测试集中可能出现的未见过的类别（映射为特殊值）
  
- **缺失值处理**:
  - 检测"?"作为缺失值标记
  - 数值特征: 使用训练集均值填充
  - 分类特征: 使用训练集众数填充
  
- **特征标准化**:
  - 使用Z-score标准化: x' = (x - μ) / σ
  - 在训练集上计算均值μ和标准差σ
  - 使用相同的参数标准化测试集
  
- **标签编码**:
  - "<=50K" → -1
  - ">50K" → +1

### 3. SVM分类器模块

**功能**: 实现SVM训练和预测

**接口**:
```python
class SVMClassifier:
    def __init__(self, learning_rate: float = 0.001, 
                 lambda_param: float = 0.01, 
                 n_epochs: int = 100):
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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用SGD训练SVM
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,) 值为-1或+1
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新样本
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
        
        返回:
            预测标签 (n_samples,) 值为-1或+1
        """
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        参数:
            X: 特征矩阵
            y: 真实标签
        
        返回:
            准确率 (0-1之间的浮点数)
        """
        pass
```

**实现细节**:

**训练算法（SGD）**:
```
初始化: a = 0, b = 0
对于每个epoch:
    对于每个训练样本 (x_k, y_k) (随机顺序):
        计算 margin = y_k * (a^T * x_k + b)
        
        如果 margin >= 1:  # 样本被正确分类且在边界外
            ∇_a = λ * a
            ∇_b = 0
        否则:  # 样本违反边界约束
            ∇_a = λ * a - y_k * x_k
            ∇_b = -y_k
        
        更新参数:
            a = a - η * ∇_a
            b = b - η * ∇_b
```

**预测**:
```
对于每个样本 x:
    score = a^T * x + b
    prediction = sign(score)  # +1 如果 score >= 0, 否则 -1
```

**超参数选择**:
- 学习率η: 初始值0.001，可能需要调整（0.0001 - 0.01范围）
- 训练轮数: 100-1000 epochs
- 正则化常数λ: 按要求测试[1e-3, 1e-2, 1e-1, 1]

### 4. 评估模块

**功能**: 模型性能评估

**接口**:
```python
def evaluate_model(model: SVMClassifier, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> dict:
    """
    评估模型性能
    
    返回:
        包含准确率等指标的字典
    """
    pass
```

### 5. 正则化分析模块

**功能**: 比较不同正则化常数的效果

**接口**:
```python
def regularization_analysis(X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_test: np.ndarray, 
                            y_test: np.ndarray,
                            lambda_values: list) -> dict:
    """
    对不同λ值进行实验
    
    参数:
        lambda_values: 要测试的λ值列表
    
    返回:
        {lambda: accuracy} 字典
    """
    pass
```

### 6. 可视化模块（可选）

**功能**: 生成结果图表

**接口**:
```python
def plot_regularization_results(results: dict, 
                                save_path: str) -> None:
    """
    绘制λ vs 准确率图表
    """
    pass
```

## 数据模型

### 输入数据格式

**训练数据** (`train.txt`):
```
age, workclass, fnlwgt, education, education-num, marital-status, 
occupation, relationship, race, sex, capital-gain, capital-loss, 
hours-per-week, native-country, income-label
```

**测试数据** (`test.txt`):
```
age, workclass, fnlwgt, education, education-num, marital-status, 
occupation, relationship, race, sex, capital-gain, capital-loss, 
hours-per-week, native-country
```

### 内部数据表示

- **特征矩阵 X**: numpy数组，形状 (n_samples, n_features)
- **标签向量 y**: numpy数组，形状 (n_samples,)，值为-1或+1
- **权重向量 a**: numpy数组，形状 (n_features,)
- **偏置 b**: 标量

### 输出数据格式

**结果摘要文件** (`results_summary.txt`):
```
SVM分类器训练结果
==================

数据集信息:
- 训练样本数: 43957
- 测试样本数: 4885
- 特征维度: [计算得出]

正则化分析结果:
λ = 0.001: 准确率 = XX.XX%
λ = 0.010: 准确率 = XX.XX%
λ = 0.100: 准确率 = XX.XX%
λ = 1.000: 准确率 = XX.XX%

分析说明:
[关于正则化如何影响准确率的解释]
```

## 错误处理

### 数据加载错误
- **问题**: 文件不存在或格式错误
- **处理**: 打印清晰的错误信息并退出程序

### 数值稳定性
- **问题**: 梯度爆炸或消失
- **处理**: 
  - 使用适当的学习率
  - 监控训练过程中的权重范数
  - 必要时进行梯度裁剪

### 缺失值
- **问题**: 数据中包含"?"
- **处理**: 使用训练集统计信息进行插补

### 未见过的类别
- **问题**: 测试集中出现训练集没有的分类值
- **处理**: 映射到特殊的"未知"类别编码

## 测试策略

### 单元测试
1. **数据加载测试**: 验证能正确读取和解析数据
2. **预处理测试**: 验证特征编码和标准化的正确性
3. **梯度计算测试**: 验证梯度公式实现正确
4. **预测测试**: 验证sign函数和预测逻辑

### 集成测试
1. **端到端测试**: 从数据加载到结果输出的完整流程
2. **正则化分析测试**: 验证能正确运行多个λ值的实验

### 验证方法
1. **数学验证**: 手动计算小样本的梯度，与代码输出对比
2. **收敛验证**: 观察训练损失是否下降
3. **合理性检查**: 准确率应在合理范围内（70%-85%）

## 性能考虑

### 时间复杂度
- **训练**: O(n_epochs × n_samples × n_features)
- **预测**: O(n_samples × n_features)

### 空间复杂度
- **存储**: O(n_samples × n_features) 用于数据矩阵
- **模型**: O(n_features) 用于权重向量

### 优化策略
1. **向量化操作**: 使用NumPy的向量化操作而非Python循环
2. **批量预测**: 一次性预测所有测试样本
3. **内存管理**: 避免不必要的数据复制

## 实现顺序

1. **阶段1**: 数据加载和基本预处理
2. **阶段2**: 完整的特征工程（编码、标准化）
3. **阶段3**: SVM核心算法实现（SGD训练）
4. **阶段4**: 预测和评估功能
5. **阶段5**: 正则化分析
6. **阶段6**: 结果输出和可视化
7. **阶段7**: 代码文档和注释完善

## 关键设计决策

### 1. 为什么使用标签编码而非one-hot编码？
- 标签编码更简单，特征维度更低
- 对于SVM，标签编码通常足够有效
- 减少计算复杂度

### 2. 学习率调整策略
- 初始使用固定学习率
- 如果不收敛，可以实现学习率衰减: η_t = η_0 / (1 + decay × t)

### 3. 随机化策略
- 每个epoch开始时打乱训练数据顺序
- 使用固定随机种子以保证可重复性

### 4. 正则化的作用
- λ较小: 模型更复杂，可能过拟合，训练准确率高但测试准确率可能较低
- λ较大: 模型更简单，防止过拟合，但可能欠拟合
- 需要找到平衡点
