# 设计文档

## 概述

本项目实现线性回归和逻辑回归算法，用于完成机器学习作业。系统将使用Python实现，利用NumPy进行数值计算，使用Matplotlib进行可视化。设计采用模块化架构，将数据加载、模型训练、评估和可视化分离为独立组件。

## 架构

### 系统架构

```
ml-regression-homework/
├── linear_regression.py      # 线性回归实现
├── logistic_regression.py    # 逻辑回归实现
├── utils.py                   # 工具函数（数据加载、误差计算等）
├── visualization.py           # 可视化函数
├── exercise_one.py           # 练习一的主程序
├── exercise_two.py           # 练习二的主程序
└── analysis/                 # 存储生成的图表和结果
    ├── linear_gd_errors.png
    ├── linear_sgd_errors.png
    ├── logistic_convergence.png
    └── logistic_training_size.png
```

### 技术栈

- **Python 3.x**: 主要编程语言
- **NumPy**: 数值计算和矩阵运算
- **Matplotlib**: 数据可视化
- **标准库**: random（随机采样）、os（文件操作）

## 组件和接口

### 1. 数据加载模块 (utils.py)

#### 函数: `load_data(filename)`
```python
def load_data(filename):
    """
    加载数据文件
    
    参数:
        filename: 数据文件路径
    
    返回:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
    """
```

#### 函数: `add_intercept(X)`
```python
def add_intercept(X):
    """
    为特征矩阵添加截距项（全1列）
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
    
    返回:
        X_with_intercept: (n_samples, n_features + 1)
    """
```

### 2. 线性回归模块 (linear_regression.py)

#### 类: `LinearRegression`

**属性:**
- `weights`: 参数向量 (包含w0, w1, ..., wp)
- `learning_rate`: 学习率
- `n_iterations`: 迭代次数
- `training_errors`: 训练误差历史
- `testing_errors`: 测试误差历史

**方法:**

```python
def __init__(self, learning_rate=0.00015, n_iterations=1500000):
    """初始化模型参数"""

def predict(self, X):
    """
    预测函数
    
    参数:
        X: 特征矩阵 (n_samples, n_features + 1) 包含截距项
    
    返回:
        predictions: 预测值 (n_samples,)
    """

def compute_mse(self, X, y):
    """
    计算均方误差
    
    参数:
        X: 特征矩阵
        y: 真实标签
    
    返回:
        mse: 均方误差值
    """

def gradient_descent(self, X_train, y_train, X_test, y_test, record_interval=100000):
    """
    批量梯度下降训练
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签
        X_test: 测试特征矩阵
        y_test: 测试标签
        record_interval: 记录误差的间隔
    
    返回:
        weights: 最优参数
        training_errors: 训练误差列表
        testing_errors: 测试误差列表
        iterations: 记录误差的迭代次数列表
    """

def stochastic_gradient_descent(self, X_train, y_train, X_test, y_test, 
                                 n_iterations=100000, record_interval=1000):
    """
    随机梯度下降训练
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签
        X_test: 测试特征矩阵
        y_test: 测试标签
        n_iterations: 迭代次数
        record_interval: 记录误差的间隔
    
    返回:
        weights: 最优参数
        training_errors: 训练误差列表
        testing_errors: 测试误差列表
        iterations: 记录误差的迭代次数列表
    """
```

**梯度下降算法:**

对于线性回归，损失函数为均方误差（MSE）：

$$J(w) = \frac{1}{2n}\sum_{i=1}^{n}(h_w(x^{(i)}) - y^{(i)})^2$$

其中 $h_w(x) = w^T x = w_0 + w_1x_1 + ... + w_px_p$

梯度为：

$$\frac{\partial J}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n}(h_w(x^{(i)}) - y^{(i)})x_j^{(i)}$$

参数更新规则：

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

### 3. 逻辑回归模块 (logistic_regression.py)

#### 类: `LogisticRegression`

**属性:**
- `weights`: 参数向量
- `learning_rate`: 学习率
- `n_iterations`: 迭代次数
- `objective_values`: 目标函数值历史

**方法:**

```python
def __init__(self, learning_rate=0.01, n_iterations=10000):
    """初始化模型参数"""

def sigmoid(self, z):
    """
    Sigmoid激活函数
    
    参数:
        z: 输入值
    
    返回:
        sigmoid(z): 1 / (1 + exp(-z))
    """

def predict_proba(self, X):
    """
    预测概率
    
    参数:
        X: 特征矩阵
    
    返回:
        probabilities: 预测为类别1的概率
    """

def predict(self, X, threshold=0.5):
    """
    预测类别标签
    
    参数:
        X: 特征矩阵
        threshold: 分类阈值
    
    返回:
        predictions: 预测标签 (0或1)
    """

def compute_log_likelihood(self, X, y):
    """
    计算条件对数似然
    
    参数:
        X: 特征矩阵
        y: 真实标签
    
    返回:
        log_likelihood: 对数似然值
    """

def stochastic_gradient_ascent(self, X_train, y_train):
    """
    随机梯度上升训练
    
    参数:
        X_train: 训练特征矩阵
        y_train: 训练标签
    
    返回:
        weights: 最优参数
        objective_values: 目标函数值列表
    """

def evaluate(self, X, y):
    """
    评估模型性能
    
    参数:
        X: 特征矩阵
        y: 真实标签
    
    返回:
        n_misclassified: 误分类样本数
        accuracy: 准确率
    """
```

**逻辑回归数学推导:**

目标函数（条件对数似然）：

$$\ell(w) = \sum_{i=1}^{n}[y^{(i)}\log(h_w(x^{(i)})) + (1-y^{(i)})\log(1-h_w(x^{(i)}))]$$

其中 $h_w(x) = \sigma(w^T x) = \frac{1}{1+e^{-w^T x}}$

梯度（偏导数）：

$$\frac{\partial \ell}{\partial w_0} = \sum_{i=1}^{n}(y^{(i)} - h_w(x^{(i)}))$$

$$\frac{\partial \ell}{\partial w_j} = \sum_{i=1}^{n}(y^{(i)} - h_w(x^{(i)}))x_j^{(i)}$$

参数更新规则（梯度上升）：

$$w_j := w_j + \alpha \frac{\partial \ell}{\partial w_j}$$

### 4. 可视化模块 (visualization.py)

#### 函数: `plot_errors(iterations, train_errors, test_errors, title, filename)`
```python
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
```

#### 函数: `plot_objective(iterations, objective_values, title, filename)`
```python
def plot_objective(iterations, objective_values, title, filename):
    """
    绘制目标函数值曲线
    
    参数:
        iterations: 迭代次数列表
        objective_values: 目标函数值列表
        title: 图表标题
        filename: 保存文件名
    """
```

#### 函数: `plot_training_size_analysis(sizes, train_errors, test_errors, filename)`
```python
def plot_training_size_analysis(sizes, train_errors, test_errors, filename):
    """
    绘制训练集大小分析图
    
    参数:
        sizes: 训练集大小列表
        train_errors: 训练误差列表
        test_errors: 测试误差列表
        filename: 保存文件名
    """
```

### 5. 主程序模块

#### exercise_one.py

执行线性回归的所有实验：
1. 使用梯度下降训练（学习率0.00015）
2. 使用不同学习率训练（学习率0.0002）
3. 使用随机梯度下降训练
4. 生成所有相关图表和分析

#### exercise_two.py

执行逻辑回归的所有实验：
1. 推导并输出数学公式
2. 训练逻辑回归模型
3. 评估测试集性能
4. 绘制收敛曲线
5. 执行训练集大小分析

## 数据模型

### 线性回归数据格式

**训练数据 (dataForTrainingLinear.txt):**
- 每行一个样本
- 列1: 房屋面积（平方米）
- 列2: 距离（千米）
- 列3: 价格（亿元）
- 样本数: 50

**测试数据 (dataForTestingLinear.txt):**
- 格式同训练数据

### 逻辑回归数据格式

**训练数据 (dataForTrainingLogistic.txt):**
- 每行一个样本
- 列1-6: 特征
- 列7: 标签（0或1）
- 样本数: 400

**测试数据 (dataForTestingLogistic.txt):**
- 格式同训练数据

## 错误处理

### 数据加载错误
- 检查文件是否存在
- 验证数据格式是否正确
- 处理空值或异常值

### 数值稳定性
- Sigmoid函数：处理极大或极小的输入值，防止溢出
- 对数计算：添加小常数避免log(0)
- 梯度检查：确保梯度不会过大导致发散

### 收敛问题
- 监控损失函数/目标函数的变化
- 如果发散，提示调整学习率
- 设置最大迭代次数防止无限循环

## 测试策略

### 单元测试

1. **数据加载测试**
   - 验证加载的数据维度正确
   - 验证特征和标签分离正确

2. **模型预测测试**
   - 测试线性回归预测函数
   - 测试逻辑回归sigmoid函数
   - 测试逻辑回归预测函数

3. **误差计算测试**
   - 验证MSE计算正确性
   - 验证对数似然计算正确性

4. **梯度计算测试**
   - 使用数值梯度验证解析梯度
   - 确保梯度下降/上升方向正确

### 集成测试

1. **端到端训练测试**
   - 在小数据集上快速训练
   - 验证训练后误差下降
   - 验证参数更新正确

2. **可视化测试**
   - 验证图表生成成功
   - 检查图表包含所需元素（标签、图例等）

### 实验验证

1. **线性回归验证**
   - 训练误差应随迭代次数下降
   - 测试误差应在合理范围内
   - 不同学习率应显示不同收敛行为

2. **逻辑回归验证**
   - 目标函数应单调递增（梯度上升）
   - 分类准确率应达到合理水平
   - 训练集大小增加时，训练误差上升、测试误差下降

## 性能考虑

### 计算效率
- 使用NumPy向量化操作避免Python循环
- 批量梯度下降：一次处理所有样本
- 随机梯度下降：减少每次迭代的计算量

### 内存使用
- 数据集较小（50-400样本），内存不是瓶颈
- 避免存储所有迭代的中间结果，只在指定间隔记录

### 可扩展性
- 模块化设计便于添加新的优化算法
- 参数化配置便于调整超参数
- 可视化函数可复用于其他实验

## 实现注意事项

1. **参数初始化**: 线性回归所有参数初始化为0.0
2. **学习率选择**: 需要仔细调整以确保收敛
3. **特征缩放**: 可能需要标准化特征以加速收敛（可选）
4. **随机种子**: 为随机梯度下降和数据采样设置随机种子以确保可重复性
5. **收敛判断**: 可以通过监控连续迭代间目标函数变化来判断收敛
