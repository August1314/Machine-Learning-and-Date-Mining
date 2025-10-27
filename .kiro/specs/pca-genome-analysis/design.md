# 设计文档

## 概述

本设计文档描述了基因组数据PCA分析的技术方案。系统将加载995个个体的基因组数据，转换为二值矩阵，执行PCA分析，并生成多个可视化图表来回答作业中的7个问题。

## 架构

### 系统组件

```
Homework 5 PCA/
├── pca_genome_analysis.py      # 主程序文件
├── p4dataset2024.txt           # 基因组数据
├── p4dataset2024_decoding.txt  # 种群标签解码
├── results/                    # 输出目录
│   ├── pc1_pc2_scatter.png    # 问题(b)的图
│   ├── pc1_pc3_scatter.png    # 问题(d)的图
│   ├── pc3_values.png         # 问题(f)的图
│   └── analysis_results.txt   # 所有问题的答案
└── README.md                   # 说明文档
```

### 数据流

1. **数据加载** → 2. **二值化** → 3. **PCA分析** → 4. **可视化** → 5. **结果输出**

## 组件和接口

### 1. 数据加载模块

**功能**: 从文件加载基因组数据和元数据

**接口**:
```python
def load_genome_data(filepath: str) -> tuple:
    """
    加载基因组数据
    
    参数:
        filepath: 数据文件路径
    
    返回:
        (ids, sexes, populations, genomes)
        - ids: 个体ID列表
        - sexes: 性别列表 (M/F)
        - populations: 种群标签列表
        - genomes: 基因组数据矩阵 (995 x 10101)
    """
    pass

def load_population_decoding(filepath: str) -> dict:
    """
    加载种群标签解码
    
    参数:
        filepath: 解码文件路径
    
    返回:
        {population_code: population_name} 字典
    """
    pass
```

**实现细节**:
- 使用pandas或numpy读取文本文件
- 分离元数据（前3列）和基因数据（后10101列）
- 处理可能的空格和格式问题

### 2. 数据预处理模块

**功能**: 将核苷酸序列转换为二值矩阵

**接口**:
```python
def compute_mode_nucleobases(genomes: np.ndarray) -> np.ndarray:
    """
    计算每个位置的众数核苷酸
    
    参数:
        genomes: 原始基因组数据 (995 x 10101)
    
    返回:
        mode_bases: 每个位置的众数核苷酸 (10101,)
    """
    pass

def create_binary_matrix(genomes: np.ndarray, mode_bases: np.ndarray) -> np.ndarray:
    """
    创建二值矩阵
    
    参数:
        genomes: 原始基因组数据
        mode_bases: 众数核苷酸
    
    返回:
        X: 二值矩阵 (995 x 10101)
            X[i,j] = 0 如果个体i在位置j有众数核苷酸
            X[i,j] = 1 否则（突变）
    """
    pass
```

**实现细节**:
- 使用scipy.stats.mode或collections.Counter计算众数
- 向量化操作以提高性能
- 处理可能的平局情况（多个众数）

### 3. PCA分析模块

**功能**: 执行主成分分析

**接口**:
```python
class PCAAnalyzer:
    def __init__(self, n_components: int = 3):
        """
        初始化PCA分析器
        
        参数:
            n_components: 要计算的主成分数量
        """
        self.n_components = n_components
        self.pca = None
        self.X_centered = None
        self.X_projected = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合PCA并转换数据
        
        参数:
            X: 二值矩阵 (995 x 10101)
        
        返回:
            X_projected: 投影后的数据 (995 x n_components)
        """
        pass
    
    def get_components(self) -> np.ndarray:
        """
        获取主成分向量
        
        返回:
            components: 主成分矩阵 (n_components x 10101)
        """
        pass
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        获取解释方差比例
        
        返回:
            explained_variance_ratio: 每个主成分解释的方差比例
        """
        pass
```

**实现细节**:
- 使用sklearn.decomposition.PCA
- 设置center=True, scale=False（中心化但不标准化）
- 确保主成分是单位向量

### 4. 可视化模块

**功能**: 生成各种散点图和分析图表

**接口**:
```python
def plot_pc1_pc2_scatter(X_projected: np.ndarray, 
                        populations: list, 
                        pop_decoding: dict,
                        save_path: str) -> None:
    """
    绘制PC1 vs PC2散点图（问题b）
    
    参数:
        X_projected: 投影数据
        populations: 种群标签
        pop_decoding: 种群解码字典
        save_path: 保存路径
    """
    pass

def plot_pc1_pc3_scatter(X_projected: np.ndarray,
                        labels: list,
                        label_name: str,
                        save_path: str) -> None:
    """
    绘制PC1 vs PC3散点图（问题d）
    
    参数:
        X_projected: 投影数据
        labels: 标签列表（性别或种群）
        label_name: 标签名称
        save_path: 保存路径
    """
    pass

def plot_pc3_values(pc3: np.ndarray, save_path: str) -> None:
    """
    绘制PC3值分布图（问题f）
    
    参数:
        pc3: 第三主成分向量 (10101,)
        save_path: 保存路径
    """
    pass
```

**实现细节**:
- 使用matplotlib创建高质量图表
- 为不同种群/标签使用不同颜色
- 添加清晰的图例、轴标签和标题
- 保存为高分辨率PNG（300 DPI）

### 5. 分析模块

**功能**: 回答作业中的各个问题

**接口**:
```python
def answer_question_a(X: np.ndarray) -> str:
    """回答问题(a): 主成分维度"""
    pass

def answer_question_c(X_projected: np.ndarray, populations: list) -> str:
    """回答问题(c): 解释PC1和PC2"""
    pass

def answer_question_e(X_projected: np.ndarray, sexes: list) -> str:
    """回答问题(e): 解释PC3"""
    pass

def answer_question_f(pc3: np.ndarray) -> str:
    """回答问题(f): 分析PC3值分布"""
    pass

def answer_question_g(pca: PCAAnalyzer, X: np.ndarray) -> str:
    """回答问题(g): 方差解释比例"""
    pass
```

### 6. 结果输出模块

**功能**: 保存所有分析结果

**接口**:
```python
def save_results(answers: dict, save_path: str) -> None:
    """
    保存所有问题的答案
    
    参数:
        answers: {question: answer} 字典
        save_path: 保存路径
    """
    pass
```

## 数据模型

### 输入数据格式

**p4dataset2024.txt**:
```
ID    Sex    Population    Nucleobase1    Nucleobase2    ...    Nucleobase10101
HG00096    M    GBR    A    T    G    ...    C
HG00097    F    GBR    A    T    G    ...    C
...
```

**p4dataset2024_decoding.txt**:
```
GBR: British in England and Scotland
FIN: Finnish in Finland
...
```

### 内部数据表示

- **原始基因组**: numpy数组 (995, 10101)，元素为字符串 'A', 'T', 'G', 'C'
- **二值矩阵X**: numpy数组 (995, 10101)，元素为0或1
- **主成分**: numpy数组 (n_components, 10101)
- **投影数据**: numpy数组 (995, n_components)

### 输出数据格式

**analysis_results.txt**:
```
PCA基因组分析结果
==================

问题(a): 主成分维度
答案: 10101
解释: ...

问题(b): PC1 vs PC2散点图
图表: results/pc1_pc2_scatter.png

问题(c): 解释PC1和PC2
答案: ...

...

问题(g): 方差解释比例
PC1解释的方差: XX.XX%
前3个PC解释的方差: XX.XX%
讨论: ...
```

## 算法详解

### PCA算法步骤

1. **中心化数据**:
   ```
   X_centered = X - mean(X, axis=0)
   ```

2. **计算协方差矩阵**:
   ```
   Cov = (1/n) * X_centered^T * X_centered
   ```

3. **特征值分解**:
   ```
   eigenvalues, eigenvectors = eig(Cov)
   ```

4. **选择主成分**:
   - 按特征值降序排列
   - 选择前k个特征向量作为主成分

5. **投影数据**:
   ```
   X_projected = X_centered * eigenvectors[:, :k]
   ```

### 二值化算法

```python
for j in range(10101):
    # 计算第j列的众数
    mode_base = most_common(X[:, j])
    
    # 创建二值列
    for i in range(995):
        if X[i, j] == mode_base:
            binary_X[i, j] = 0
        else:
            binary_X[i, j] = 1
```

## 错误处理

### 数据加载错误
- **问题**: 文件不存在或格式错误
- **处理**: 打印清晰错误信息并退出

### 数据格式错误
- **问题**: 核苷酸不是A/T/G/C
- **处理**: 记录警告，可能将其视为缺失值

### PCA计算错误
- **问题**: 矩阵奇异或数值不稳定
- **处理**: 使用sklearn的稳定实现

## 性能考虑

### 时间复杂度
- **数据加载**: O(n × m) = O(995 × 10101)
- **二值化**: O(n × m)
- **PCA**: O(min(n², m²) × min(n, m)) ≈ O(995² × 995)
- **可视化**: O(n) = O(995)

### 空间复杂度
- **原始数据**: O(n × m) ≈ 10MB
- **二值矩阵**: O(n × m) ≈ 10MB
- **主成分**: O(k × m) ≈ 30KB (k=3)

### 优化策略
1. 使用numpy向量化操作
2. 只计算需要的主成分数量（k=3）
3. 使用sklearn的高效实现

## 测试策略

### 单元测试
1. **数据加载测试**: 验证能正确读取和解析数据
2. **二值化测试**: 验证众数计算和二值转换正确
3. **PCA测试**: 验证主成分是单位向量，正交

### 集成测试
1. **端到端测试**: 从数据加载到结果输出的完整流程
2. **可视化测试**: 验证图表能正确生成和保存

### 验证方法
1. **维度检查**: 确保所有矩阵维度正确
2. **数值检查**: 验证主成分是单位向量
3. **可视化检查**: 手动检查图表是否合理

## 实现顺序

1. **阶段1**: 数据加载和基本预处理
2. **阶段2**: 二值矩阵转换
3. **阶段3**: PCA分析实现
4. **阶段4**: 问题(a)(b)(c)实现
5. **阶段5**: 问题(d)(e)实现
6. **阶段6**: 问题(f)(g)实现
7. **阶段7**: 结果整理和文档

## 关键设计决策

### 1. 为什么使用sklearn的PCA？
- 经过充分测试和优化
- 处理数值稳定性问题
- 符合作业要求（允许使用库实现）

### 2. 如何处理众数平局？
- 使用第一个出现的众数
- 或使用字母顺序（A < C < G < T）

### 3. 图表配色方案
- 使用matplotlib的tab10或Set3调色板
- 确保颜色区分度高
- 考虑色盲友好

### 4. 中文支持
- 设置matplotlib中文字体
- 确保图表标签清晰可读

## 生物学背景

### 主成分的可能解释

**PC1和PC2**: 
- 可能捕捉地理和历史迁移模式
- 反映种群的遗传距离
- 与人类迁出非洲的历史相关

**PC3**:
- 可能与性别相关（X/Y染色体）
- 可能捕捉特定染色体的变异

### 染色体相关
- 人类有23对染色体
- X染色体和Y染色体决定性别
- PC3的值分布可能显示染色体边界
