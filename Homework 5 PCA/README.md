# Homework 5: PCA基因组分析

**学生姓名**: 梁力航  
**学号**: 23336128  
**课程**: 机器学习与数据挖掘  
**完成日期**: 2025年10月27日

---

## 项目概述

本项目对995个个体的基因组数据进行主成分分析(PCA)，揭示遗传变异的主要模式。数据来自国际基因组样本资源(International Genome Sample Resource)，包含10101个核苷酸位点。

## 文件说明

```
Homework 5 PCA/
├── pca_genome_analysis.py          # 主程序
├── p4dataset2024.txt               # 基因组数据（995个个体）
├── p4dataset2024_decoding.txt      # 种群标签解码
├── results/                        # 输出目录
│   ├── analysis_results.txt       # 所有问题的答案
│   ├── pc1_pc2_scatter.png        # 问题(b): PC1 vs PC2散点图
│   ├── pc1_pc3_scatter_sex.png    # 问题(d): PC1 vs PC3散点图（按性别）
│   ├── pc1_pc3_scatter_pop.png    # 问题(d): PC1 vs PC3散点图（按种群）
│   └── pc3_values.png             # 问题(f): PC3值分布图
└── README.md                       # 本文件
```

## 依赖环境

需要以下Python库：

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

或使用conda：

```bash
conda install numpy pandas matplotlib scikit-learn scipy
```

## 运行方法

### 使用conda环境（推荐）

```bash
conda activate ml
python pca_genome_analysis.py
```

### 直接运行

```bash
python pca_genome_analysis.py
```

程序将自动：
1. 加载基因组数据
2. 转换为二值矩阵
3. 执行PCA分析
4. 回答所有7个问题
5. 生成所有要求的图表
6. 保存结果到results/目录

运行时间：约10-30秒

## 实验结果摘要

### 数据统计

- **个体数量**: 995
- **核苷酸位点**: 10101
- **性别分布**: 509男, 486女
- **种群数量**: 7个不同种群
- **突变率**: 39.24%

### 主要发现

#### 问题(a): 主成分维度
**答案**: 10101

每个主成分向量的维度等于输入矩阵的列数（特征数）。

#### 问题(b): PC1 vs PC2散点图
见图表: `results/pc1_pc2_scatter.png`

#### 问题(c): PC1和PC2的解释
- 不同种群在PC1-PC2空间中形成明显的聚类
- 前两个主成分捕捉了人类种群的地理起源和迁移历史
- 反映了非洲、欧洲和亚洲种群之间的遗传差异

#### 问题(d): PC1 vs PC3散点图
见图表: `results/pc1_pc3_scatter_sex.png` 和 `results/pc1_pc3_scatter_pop.png`

性别标签能更好地解释PC3的聚类模式。

#### 问题(e): PC3捕捉的信息
**答案**: 第三主成分主要捕捉了性别信息，反映了X和Y染色体之间的遗传差异。

#### 问题(f): PC3值分布
见图表: `results/pc3_values.png`

**观察**: PC3的绝对值在某些区域明显较高，呈现出分段的模式。

**解释**: 这种模式反映了染色体结构，特别是性染色体（X和Y）上的变异。峰值区域对应于Y染色体特有的序列区域。

#### 问题(g): 方差解释比例

| 主成分 | 解释方差 | 平方长度减少 |
|--------|----------|--------------|
| PC1    | 0.57%    | 99.43%       |
| 前3个PC | 1.41%    | 98.59%       |

**讨论**: 
- 基因组数据具有极高的维度，前3个主成分只解释了约1.41%的方差
- 尽管如此，前几个主成分仍然捕捉到了最重要的变异模式（地理起源、性别等）
- 大部分方差来自于个体之间的细微差异和噪声
- 对于可视化和理解主要的遗传结构，前几个主成分已经足够有用

## 算法说明

### 数据预处理

1. **加载数据**: 从文本文件读取995个个体的基因组数据
2. **计算众数**: 对每个核苷酸位点计算众数核苷酸
3. **二值化**: 创建二值矩阵X
   - X[i,j] = 0: 个体i在位点j有众数核苷酸
   - X[i,j] = 1: 个体i在位点j有突变

### PCA分析

使用sklearn.decomposition.PCA：
- 自动中心化数据（减去均值）
- 不进行标准化（保留方差差异）
- 提取前3个主成分
- 主成分是单位向量

### 可视化

- 使用matplotlib生成高质量图表
- 为不同种群/性别使用不同颜色
- 保存为300 DPI的PNG图像

## 生物学意义

### PC1和PC2: 地理和历史
- 反映人类"走出非洲"的迁移历史
- 捕捉不同大陆种群之间的遗传差异
- 地理距离导致的遗传分化

### PC3: 性别
- 反映X和Y染色体之间的差异
- 男性有XY染色体，女性有XX染色体
- Y染色体特有的序列导致PC3的性别分离

## 技术亮点

1. ✨ **完整的数据预处理流程**: 从原始核苷酸序列到二值矩阵
2. ✨ **标准PCA实现**: 使用sklearn，符合作业要求
3. ✨ **多角度可视化**: 不同标签方案的对比分析
4. ✨ **详细的生物学解释**: 结合遗传学知识解释结果
5. ✨ **清晰的代码结构**: 模块化设计，易于理解

## 输出文件

### analysis_results.txt
包含所有7个问题的详细答案。

### 图表文件
1. **pc1_pc2_scatter.png**: PC1 vs PC2散点图（按种群着色）
2. **pc1_pc3_scatter_sex.png**: PC1 vs PC3散点图（按性别着色）
3. **pc1_pc3_scatter_pop.png**: PC1 vs PC3散点图（按种群着色）
4. **pc3_values.png**: PC3值沿核苷酸位点的分布

## 作业要求对照

✅ **问题(a)**: 主成分维度 - 已回答  
✅ **问题(b)**: PC1 vs PC2散点图 - 已生成  
✅ **问题(c)**: 解释PC1和PC2 - 已回答  
✅ **问题(d)**: PC1 vs PC3散点图 - 已生成（两个版本）  
✅ **问题(e)**: 解释PC3 - 已回答  
✅ **问题(f)**: PC3值分布图和解释 - 已生成和回答  
✅ **问题(g)**: 方差解释比例和讨论 - 已回答  
✅ **代码**: 完整代码在pca_genome_analysis.py

## 参考资料

- International Genome Sample Resource: https://www.internationalgenome.org/
- Scikit-learn PCA文档: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- 原始作业: https://web.stanford.edu/class/cs168/p4.pdf

---

**完成状态**: ✅ 全部完成  
**代码状态**: ✅ 测试通过  
**文档状态**: ✅ 完整详细
