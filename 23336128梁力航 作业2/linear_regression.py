"""
线性回归模型实现
使用梯度下降和随机梯度下降优化
"""
import numpy as np


class LinearRegression:
    """线性回归模型类"""
    
    def __init__(self, learning_rate=0.00015, n_iterations=1500000):
        """
        初始化模型参数
        
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.training_errors = []
        self.testing_errors = []
    
    def predict(self, X):
        """
        预测函数
        
        参数:
            X: 特征矩阵 (n_samples, n_features + 1) 包含截距项
        
        返回:
            predictions: 预测值 (n_samples,)
        """
        return np.dot(X, self.weights)
    
    def compute_mse(self, X, y):
        """
        计算均方误差
        
        参数:
            X: 特征矩阵
            y: 真实标签
        
        返回:
            mse: 均方误差值
        """
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse
    
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
        n_samples, n_features = X_train.shape
        
        # 初始化参数为0
        self.weights = np.zeros(n_features)
        
        training_errors = []
        testing_errors = []
        iterations = []
        
        for i in range(1, self.n_iterations + 1):
            # 计算预测值
            predictions = self.predict(X_train)
            
            # 计算梯度
            errors = predictions - y_train
            gradient = (1 / n_samples) * np.dot(X_train.T, errors)
            
            # 更新参数
            self.weights -= self.learning_rate * gradient
            
            # 记录误差
            if i % record_interval == 0:
                train_error = self.compute_mse(X_train, y_train)
                test_error = self.compute_mse(X_test, y_test)
                training_errors.append(train_error)
                testing_errors.append(test_error)
                iterations.append(i)
                print(f"迭代 {i}: 训练误差 = {train_error:.6f}, 测试误差 = {test_error:.6f}")
        
        self.training_errors = training_errors
        self.testing_errors = testing_errors
        
        return self.weights, training_errors, testing_errors, iterations
    
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
        n_samples, n_features = X_train.shape
        
        # 初始化参数为0
        self.weights = np.zeros(n_features)
        
        training_errors = []
        testing_errors = []
        iterations = []
        
        for i in range(1, n_iterations + 1):
            # 随机选择一个样本
            idx = np.random.randint(0, n_samples)
            X_i = X_train[idx:idx+1]
            y_i = y_train[idx:idx+1]
            
            # 计算预测值
            prediction = self.predict(X_i)
            
            # 计算梯度
            error = prediction - y_i
            gradient = np.dot(X_i.T, error).flatten()
            
            # 更新参数
            self.weights -= self.learning_rate * gradient
            
            # 记录误差
            if i % record_interval == 0:
                train_error = self.compute_mse(X_train, y_train)
                test_error = self.compute_mse(X_test, y_test)
                training_errors.append(train_error)
                testing_errors.append(test_error)
                iterations.append(i)
                print(f"迭代 {i}: 训练误差 = {train_error:.6f}, 测试误差 = {test_error:.6f}")
        
        return self.weights, training_errors, testing_errors, iterations
