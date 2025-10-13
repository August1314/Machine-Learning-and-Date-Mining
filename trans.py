import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 设置中文字体和随机种子
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

def demo_feature_space_transformation():
    """
    演示特征空间变换的概念
    """
    print("=== 特征空间变换演示 ===\n")
    
    # 1. 生成示例数据
    print("1. 生成原始数据")
    n_samples = 100
    X_original = np.linspace(-3, 3, n_samples).reshape(-1, 1)  # 一维特征 (p=1)
    y_true = np.sin(X_original.flatten()) + 0.1 * np.random.randn(n_samples)
    
    print(f"原始特征维度: {X_original.shape}")
    print(f"原始特征示例: {X_original[:5].flatten()}")
    
    # 2. 定义不同的特征变换方法
    def polynomial_transform(X, degree=3):
        """多项式变换: φ(x) = {1, x, x², ..., x^k}"""
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        return poly.fit_transform(X)
    
    def radial_basis_transform(X, centers=[-2, 0, 2], gamma=1.0):
        """径向基函数变换 (RBF)"""
        X_transformed = np.ones((X.shape[0], len(centers) + 1))
        for i, center in enumerate(centers):
            X_transformed[:, i] = np.exp(-gamma * (X.flatten() - center) ** 2)
        return X_transformed
    
    def spline_transform(X, knots=[-2, 0, 2]):
        """简单的样条基函数变换"""
        X_transformed = np.ones((X.shape[0], len(knots) + 1))
        X_flat = X.flatten()
        for i, knot in enumerate(knots):
            X_transformed[:, i] = np.maximum(0, (X_flat - knot)) ** 3
        return X_transformed
    
    # 3. 应用不同的变换
    print("\n2. 应用特征空间变换")
    
    # 多项式变换
    X_poly = polynomial_transform(X_original, degree=5)
    print(f"多项式变换后维度 (k={X_poly.shape[1]}): {X_poly.shape}")
    print("基函数: [1, x, x², x³, x⁴, x⁵]")
    
    # 径向基函数变换
    X_rbf = radial_basis_transform(X_original)
    print(f"RBF变换后维度: {X_rbf.shape}")
    
    # 样条变换
    X_spline = spline_transform(X_original)
    print(f"样条变换后维度: {X_spline.shape}")
    
    # 4. 可视化变换结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始数据
    axes[0, 0].scatter(X_original, y_true, alpha=0.6, s=20)
    axes[0, 0].set_title('原始数据\n$X_i \in \mathbb{R}^1$')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    
    # 多项式基函数
    degrees = 5
    x_plot = np.linspace(-3, 3, 100)
    for i in range(degrees + 1):
        axes[0, 1].plot(x_plot, x_plot**i, label=f'$x^{i}$' if i > 0 else '1')
    axes[0, 1].set_title('多项式基函数\n$\phi(x)=\{1, x, x^2, ..., x^k\}$')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('x')
    
    # 径向基函数
    centers = [-2, 0, 2]
    for center in centers:
        axes[0, 2].plot(x_plot, np.exp(-(x_plot - center)**2), 
                       label=f'center={center}')
    axes[0, 2].set_title('径向基函数 (RBF)')
    axes[0, 2].legend()
    axes[0, 2].set_xlabel('x')
    
    # 5. 使用变换后的特征进行回归
    print("\n3. 使用变换特征进行模型训练")
    
    # 原始特征（线性模型）
    model_linear = LinearRegression()
    model_linear.fit(X_original, y_true)
    y_pred_linear = model_linear.predict(X_original)
    
    # 多项式特征
    model_poly = Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('linear', LinearRegression())
    ])
    model_poly.fit(X_original, y_true)
    y_pred_poly = model_poly.predict(X_original)
    
    # 比较结果
    mse_linear = mean_squared_error(y_true, y_pred_linear)
    mse_poly = mean_squared_error(y_true, y_pred_poly)
    
    print(f"线性模型MSE: {mse_linear:.4f}")
    print(f"多项式模型MSE: {mse_poly:.4f}")
    
    # 可视化拟合结果
    x_plot = np.linspace(-3, 3, 300).reshape(-1, 1)
    
    axes[1, 0].scatter(X_original, y_true, alpha=0.6, s=20, label='数据点')
    axes[1, 0].plot(x_plot, model_linear.predict(x_plot), 'r-', 
                   label='线性拟合', linewidth=2)
    axes[1, 0].set_title('原始特征空间拟合\n(线性模型)')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    
    axes[1, 1].scatter(X_original, y_true, alpha=0.6, s=20, label='数据点')
    axes[1, 1].plot(x_plot, model_poly.predict(x_plot), 'g-', 
                   label='多项式拟合', linewidth=2)
    axes[1, 1].set_title('变换特征空间拟合\n(5次多项式)')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    
    # 特征空间的可视化（使用前两个主要成分）
    from sklearn.decomposition import PCA
    
    # 应用PCA到多项式特征
    pca = PCA(n_components=2)
    X_poly_pca = pca.fit_transform(X_poly[:, 1:])  # 去掉偏置项
    
    scatter = axes[1, 2].scatter(X_poly_pca[:, 0], X_poly_pca[:, 1], 
                               c=y_true, cmap='viridis', alpha=0.7)
    axes[1, 2].set_title('变换后特征空间的PCA可视化\n(前两个主成分)')
    axes[1, 2].set_xlabel('主成分1')
    axes[1, 2].set_ylabel('主成分2')
    plt.colorbar(scatter, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    # 6. 展示变换矩阵
    print("\n4. 特征变换示例（前5个样本）:")
    print("原始特征 X:")
    print(X_original[:5].flatten())
    
    print("\n多项式变换后 φ(X):")
    print("形状:", X_poly[:5, :].shape)
    print("数值:")
    print(X_poly[:5, :])
    
    return {
        'original_features': X_original,
        'polynomial_features': X_poly,
        'rbf_features': X_rbf,
        'spline_features': X_spline,
        'models': {
            'linear': model_linear,
            'polynomial': model_poly
        }
    }

# 运行演示
if __name__ == "__main__":
    results = demo_feature_space_transformation()