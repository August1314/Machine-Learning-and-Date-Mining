"""
逻辑回归模型实现
使用随机梯度上升优化
"""
import numpy as np


class LogisticRegression:
    """逻辑回归模型类"""
    
    def __init__(self, learning_rate=0.01, n_iterations=10000):
        """
        初始化模型参数
        
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.objective_values = []
    
    def sigmoid(self, z):
        """
        Sigmoid激活函数
        
        参数:
            z: 输入值
        
        返回:
            sigmoid(z): 1 / (1 + exp(-z))
        """
        # 数值稳定性处理
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
            X: 特征矩阵
        
        返回:
            probabilities: 预测为类别1的概率
        """
        z = np.dot(X, self.weights)
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别标签
        
        参数:
            X: 特征矩阵
            threshold: 分类阈值
        
        返回:
            predictions: 预测标签 (0或1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def compute_log_likelihood(self, X, y):
        """
        计算条件对数似然
        
        参数:
            X: 特征矩阵
            y: 真实标签
        
        返回:
            log_likelihood: 对数似然值
        """
        probabilities = self.predict_proba(X)
        # 添加小常数避免log(0)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        
        log_likelihood = np.sum(
            y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
        )
        return log_likelihood
    
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
        n_samples, n_features = X_train.shape
        
        # 初始化参数为0
        self.weights = np.zeros(n_features)
        
        objective_values = []
        
        for i in range(1, self.n_iterations + 1):
            # 随机选择一个样本
            idx = np.random.randint(0, n_samples)
            X_i = X_train[idx:idx+1]
            y_i = y_train[idx:idx+1]
            
            # 计算预测概率
            prob = self.predict_proba(X_i)
            
            # 计算梯度
            error = y_i - prob
            gradient = np.dot(X_i.T, error).flatten()
            
            # 更新参数（梯度上升）
            self.weights += self.learning_rate * gradient
            
            # 记录目标函数值
            log_likelihood = self.compute_log_likelihood(X_train, y_train)
            objective_values.append(log_likelihood)
            
            if i % 1000 == 0:
                print(f"迭代 {i}: 对数似然 = {log_likelihood:.4f}")
        
        self.objective_values = objective_values
        
        return self.weights, objective_values
    
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
        predictions = self.predict(X)
        n_misclassified = np.sum(predictions != y)
        accuracy = np.mean(predictions == y)
        
        return n_misclassified, accuracy
