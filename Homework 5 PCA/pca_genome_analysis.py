"""
PCA基因组分析 - 作业5
对基因组数据进行主成分分析，揭示遗传变异模式

作者: 梁力航
学号: 23336128
课程: 机器学习与数据挖掘
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import os
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('results', exist_ok=True)


def load_genome_data(filepath):
    """
    加载基因组数据
    
    参数:
        filepath: 数据文件路径
    
    返回:
        ids: 个体ID列表
        sexes: 性别列表 (M/F)
        populations: 种群标签列表
        genomes: 基因组数据矩阵 (n_individuals x 10101)
    """
    print("正在加载基因组数据...")
    
    # 读取数据文件
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            data.append(parts)
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 分离元数据和基因数据
    ids = data[:, 0]
    sexes = data[:, 1]
    populations = data[:, 2]
    genomes = data[:, 3:]  # 10101个核苷酸
    
    print(f"加载完成: {len(ids)}个个体, {genomes.shape[1]}个核苷酸位点")
    print(f"性别分布: {np.sum(sexes == 'M')}男, {np.sum(sexes == 'F')}女")
    print(f"种群数量: {len(np.unique(populations))}个不同种群")
    
    return ids, sexes, populations, genomes


def load_population_decoding(filepath):
    """
    加载种群标签解码
    
    参数:
        filepath: 解码文件路径
    
    返回:
        pop_decoding: {population_code: population_name} 字典
    """
    print("\n正在加载种群解码信息...")
    
    pop_decoding = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    code, name = parts
                    pop_decoding[code] = name
    
    print(f"加载完成: {len(pop_decoding)}个种群")
    
    return pop_decoding


print("=" * 70)
print("PCA基因组分析 - 主成分分析揭示遗传变异模式")
print("=" * 70)
print()

# 加载数据
ids, sexes, populations, genomes = load_genome_data('p4dataset2024.txt')
pop_decoding = load_population_decoding('p4dataset2024_decoding.txt')


def compute_mode_nucleobases(genomes):
    """
    计算每个位置的众数核苷酸
    
    参数:
        genomes: 原始基因组数据 (n_individuals x n_positions)
    
    返回:
        mode_bases: 每个位置的众数核苷酸 (n_positions,)
    """
    print("\n正在计算每个位置的众数核苷酸...")
    
    n_positions = genomes.shape[1]
    mode_bases = []
    
    for j in range(n_positions):
        # 获取第j列的所有核苷酸
        column = genomes[:, j]
        
        # 计算众数（最频繁出现的核苷酸）
        counter = Counter(column)
        mode_base = counter.most_common(1)[0][0]
        mode_bases.append(mode_base)
    
    mode_bases = np.array(mode_bases)
    
    print(f"计算完成: {len(mode_bases)}个位点的众数核苷酸")
    
    # 统计众数核苷酸的分布
    mode_counter = Counter(mode_bases)
    print(f"众数分布: {dict(mode_counter)}")
    
    return mode_bases


def create_binary_matrix(genomes, mode_bases):
    """
    创建二值矩阵
    
    参数:
        genomes: 原始基因组数据
        mode_bases: 众数核苷酸
    
    返回:
        X: 二值矩阵 (n_individuals x n_positions)
            X[i,j] = 0 如果个体i在位置j有众数核苷酸
            X[i,j] = 1 否则（突变）
    """
    print("\n正在创建二值矩阵...")
    
    # 创建二值矩阵
    X = (genomes != mode_bases).astype(int)
    
    print(f"二值矩阵创建完成: 形状 {X.shape}")
    print(f"突变率: {np.mean(X) * 100:.2f}% (平均每个位点的突变比例)")
    print(f"验证: 所有值都是0或1 - {np.all((X == 0) | (X == 1))}")
    
    return X


# 执行数据预处理
mode_bases = compute_mode_nucleobases(genomes)
X = create_binary_matrix(genomes, mode_bases)


class PCAAnalyzer:
    """
    PCA分析器
    对基因组二值矩阵执行主成分分析
    """
    
    def __init__(self, n_components=3):
        """
        初始化PCA分析器
        
        参数:
            n_components: 要计算的主成分数量
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.X_centered = None
        self.X_projected = None
    
    def fit_transform(self, X):
        """
        拟合PCA并转换数据
        
        参数:
            X: 二值矩阵 (n_individuals x n_positions)
        
        返回:
            X_projected: 投影后的数据 (n_individuals x n_components)
        """
        print(f"\n正在执行PCA分析 (提取前{self.n_components}个主成分)...")
        
        # 拟合并转换数据
        # sklearn的PCA会自动中心化数据
        self.X_projected = self.pca.fit_transform(X)
        
        print(f"PCA完成: 投影数据形状 {self.X_projected.shape}")
        print(f"解释方差比例: {self.pca.explained_variance_ratio_}")
        print(f"累积解释方差: {np.cumsum(self.pca.explained_variance_ratio_)}")
        
        return self.X_projected
    
    def get_components(self):
        """
        获取主成分向量
        
        返回:
            components: 主成分矩阵 (n_components x n_positions)
        """
        return self.pca.components_
    
    def get_explained_variance_ratio(self):
        """
        获取解释方差比例
        
        返回:
            explained_variance_ratio: 每个主成分解释的方差比例
        """
        return self.pca.explained_variance_ratio_
    
    def get_explained_variance(self):
        """
        获取解释方差
        
        返回:
            explained_variance: 每个主成分的方差
        """
        return self.pca.explained_variance_


# 执行PCA分析
pca_analyzer = PCAAnalyzer(n_components=3)
X_projected = pca_analyzer.fit_transform(X)


# ============================================================================
# 回答问题
# ============================================================================

def answer_question_a(X):
    """回答问题(a): 主成分维度"""
    dimension = X.shape[1]
    answer = f"""
问题(a): PCA返回向量的维度

答案: {dimension}

解释: 主成分向量的维度等于输入矩阵的列数（特征数）。在本例中，我们有10101个核苷酸位点，
因此每个主成分向量都是10101维的。每个主成分是原始特征空间中的一个方向（单位向量）。
"""
    return answer


def plot_pc1_pc2_scatter(X_projected, populations, pop_decoding, save_path):
    """绘制PC1 vs PC2散点图（问题b）"""
    print("\n正在绘制PC1 vs PC2散点图...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取唯一的种群
    unique_pops = np.unique(populations)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pops)))
    
    # 为每个种群绘制散点
    for i, pop in enumerate(unique_pops):
        mask = populations == pop
        pop_name = pop_decoding.get(pop, pop)
        ax.scatter(X_projected[mask, 0], X_projected[mask, 1], 
                  c=[colors[i]], label=pop_name, alpha=0.7, s=50)
    
    ax.set_xlabel('第一主成分 (PC1)', fontsize=12)
    ax.set_ylabel('第二主成分 (PC2)', fontsize=12)
    ax.set_title('基因组数据的前两个主成分', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


def answer_question_c():
    """回答问题(c): 解释PC1和PC2"""
    answer = """
问题(c): PC1和PC2的解释

基本事实:
1. 不同种群在PC1-PC2空间中形成明显的聚类，显示出清晰的遗传结构。
2. 种群的分布模式反映了地理和历史迁移路径。

解释:
前两个主成分捕捉了人类种群的地理起源和迁移历史。PC1和PC2主要反映了：
- 非洲、欧洲和亚洲种群之间的遗传差异
- 人类"走出非洲"的迁移历史
- 地理距离导致的遗传分化

这些主成分揭示了人类进化和迁移的历史模式，不同大陆的种群在遗传上有明显区别。
"""
    return answer


def plot_pc1_pc3_scatter(X_projected, labels, label_name, save_path):
    """绘制PC1 vs PC3散点图（问题d）"""
    print(f"\n正在绘制PC1 vs PC3散点图 (按{label_name}着色)...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取唯一的标签
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    # 为每个标签绘制散点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_projected[mask, 0], X_projected[mask, 2], 
                  c=[colors[i]], label=label, alpha=0.7, s=50)
    
    ax.set_xlabel('第一主成分 (PC1)', fontsize=12)
    ax.set_ylabel('第三主成分 (PC3)', fontsize=12)
    ax.set_title(f'PC1 vs PC3 (按{label_name}分类)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


def answer_question_e():
    """回答问题(e): 解释PC3"""
    answer = """
问题(e): PC3捕捉的信息

答案: 第三主成分主要捕捉了性别信息，反映了X和Y染色体之间的遗传差异。
"""
    return answer


def plot_pc3_values(pc3, save_path):
    """绘制PC3值分布图（问题f）"""
    print("\n正在绘制PC3值分布图...")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制PC3绝对值
    indices = np.arange(1, len(pc3) + 1)
    ax.plot(indices, np.abs(pc3), linewidth=0.5, alpha=0.7)
    
    ax.set_xlabel('核苷酸位点索引', fontsize=12)
    ax.set_ylabel('PC3绝对值', fontsize=12)
    ax.set_title('第三主成分的值分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()


def answer_question_f():
    """回答问题(f): 分析PC3值分布"""
    answer = """
问题(f): PC3值分布的观察和解释

观察: PC3的绝对值在某些区域明显较高，呈现出分段的模式，在特定位置有明显的峰值。

可能解释: 这种模式很可能反映了染色体的结构。人类有23对染色体，其中X和Y染色体决定性别。
PC3捕捉的性别差异主要来自性染色体（特别是Y染色体）上的变异。图中的峰值区域可能对应于：
1. Y染色体特有的序列区域
2. X和Y染色体之间差异最大的区域
3. 性别决定相关基因的位置

这些区域的高PC3值表明它们在男性和女性之间有显著的遗传差异。
"""
    return answer


def answer_question_g(pca_analyzer, X):
    """回答问题(g): 方差解释比例"""
    # 获取解释方差比例
    explained_var_ratio = pca_analyzer.get_explained_variance_ratio()
    
    # PC1解释的方差
    pc1_var = explained_var_ratio[0] * 100
    
    # 前3个PC解释的方差
    pc3_var = np.sum(explained_var_ratio) * 100
    
    # 计算平方长度减少百分比
    # 原始数据的总方差
    X_centered = X - np.mean(X, axis=0)
    total_var = np.sum(X_centered ** 2) / len(X)
    
    # PC1投影后的方差
    pc1_projection_var = pca_analyzer.get_explained_variance()[0]
    pc1_decrease = (1 - pc1_projection_var / total_var) * 100
    
    # 前3个PC投影后的方差
    pc3_projection_var = np.sum(pca_analyzer.get_explained_variance())
    pc3_decrease = (1 - pc3_projection_var / total_var) * 100
    
    answer = f"""
问题(g): 方差解释比例

PC1解释的方差: {pc1_var:.2f}%
前3个PC解释的方差: {pc3_var:.2f}%

平方长度减少:
- 投影到PC1后，平均平方长度减少: {pc1_decrease:.2f}%
- 投影到前3个PC后，平均平方长度减少: {pc3_decrease:.2f}%

讨论:
这个结果并不令人惊讶。基因组数据具有极高的维度（10101维），而前3个主成分只解释了约{pc3_var:.2f}%的方差。
这说明：
1. 基因组变异是高度复杂和多维的，不能简单地用少数几个方向来概括。
2. 尽管如此，前几个主成分仍然捕捉到了最重要的变异模式（地理起源、性别等）。
3. 大部分方差来自于个体之间的细微差异和噪声，这些对于理解种群结构不太重要。
4. 对于可视化和理解主要的遗传结构，前几个主成分已经足够有用。
"""
    return answer


def save_results(answers, save_path):
    """保存所有问题的答案"""
    print(f"\n正在保存结果到 {save_path}...")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PCA基因组分析结果\n")
        f.write("=" * 70 + "\n\n")
        
        for question, answer in answers.items():
            f.write(f"{question}\n")
            f.write(answer)
            f.write("\n" + "-" * 70 + "\n\n")
    
    print("结果保存完成!")


# ============================================================================
# 主程序：回答所有问题
# ============================================================================

print("\n" + "=" * 70)
print("开始回答问题")
print("=" * 70)

# 收集所有答案
answers = {}

# 问题(a)
answers['问题(a)'] = answer_question_a(X)
print(answers['问题(a)'])

# 问题(b)
plot_pc1_pc2_scatter(X_projected, populations, pop_decoding, 'results/pc1_pc2_scatter.png')
answers['问题(b)'] = "见图表: results/pc1_pc2_scatter.png"

# 问题(c)
answers['问题(c)'] = answer_question_c()
print(answers['问题(c)'])

# 问题(d) - 尝试不同标签
plot_pc1_pc3_scatter(X_projected, sexes, '性别', 'results/pc1_pc3_scatter_sex.png')
plot_pc1_pc3_scatter(X_projected, populations, '种群', 'results/pc1_pc3_scatter_pop.png')
answers['问题(d)'] = "见图表: results/pc1_pc3_scatter_sex.png 和 results/pc1_pc3_scatter_pop.png\n性别标签能更好地解释PC3的聚类模式。"

# 问题(e)
answers['问题(e)'] = answer_question_e()
print(answers['问题(e)'])

# 问题(f)
pc3 = pca_analyzer.get_components()[2]
plot_pc3_values(pc3, 'results/pc3_values.png')
answers['问题(f)'] = answer_question_f()
print(answers['问题(f)'])

# 问题(g)
answers['问题(g)'] = answer_question_g(pca_analyzer, X)
print(answers['问题(g)'])

# 保存所有结果
save_results(answers, 'results/analysis_results.txt')

print("\n" + "=" * 70)
print("分析完成！所有结果已保存到 results/ 目录")
print("=" * 70)
