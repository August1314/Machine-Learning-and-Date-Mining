"""
工具函数模块
包含数据加载和预处理函数
"""
import numpy as np
import os


def load_data(filename):
    """
    加载数据文件
    
    参数:
        filename: 数据文件路径
    
    返回:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"数据文件 {filename} 不存在")
    
    data = np.loadtxt(filename)
    X = data[:, :-1]  # 所有列除了最后一列
    y = data[:, -1]   # 最后一列
    
    return X, y


def add_intercept(X):
    """
    为特征矩阵添加截距项（全1列）
    
    参数:
        X: 特征矩阵 (n_samples, n_features)
    
    返回:
        X_with_intercept: (n_samples, n_features + 1)
    """
    n_samples = X.shape[0]
    intercept = np.ones((n_samples, 1))
    return np.hstack([intercept, X])
