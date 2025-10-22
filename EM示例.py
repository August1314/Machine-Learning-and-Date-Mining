import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

class CoinEM:
    def __init__(self, true_theta_A=0.7, true_theta_B=0.3, pi_A=0.6):
        """
        初始化硬币EM算法
        
        参数:
        true_theta_A: 硬币A的真实正面概率
        true_theta_B: 硬币B的真实正面概率  
        pi_A: 选择硬币A的先验概率
        """
        self.true_theta_A = true_theta_A
        self.true_theta_B = true_theta_B
        self.pi_A = pi_A
        self.pi_B = 1 - pi_A
        
        # 生成模拟数据
        self.generate_data()
        
    def generate_data(self):
        """生成模拟实验数据"""
        np.random.seed(42)  # 保证可重复性
        
        # 5轮实验，每轮抛10次
        self.n_rounds = 5
        self.n_tosses = 10
        
        # 真实情况：每轮使用的硬币
        self.true_coins = np.random.choice(['A', 'B'], size=self.n_rounds, p=[self.pi_A, self.pi_B])
        
        # 生成抛掷结果
        self.results = []
        for coin in self.true_coins:
            if coin == 'A':
                p_head = self.true_theta_A
            else:
                p_head = self.true_theta_B
                
            # 生成10次抛掷结果
            heads = np.random.binomial(self.n_tosses, p_head)
            tails = self.n_tosses - heads
            self.results.append((heads, tails))
        
        print("=== 真实情况 ===")
        print(f"硬币A真实概率: {self.true_theta_A:.3f}")
        print(f"硬币B真实概率: {self.true_theta_B:.3f}")
        print(f"选择硬币A概率: {self.pi_A:.3f}")
        print("\n实验数据:")
        for i, ((h, t), coin) in enumerate(zip(self.results, self.true_coins)):
            print(f"第{i+1}轮: {h}正 {t}反 (真实使用硬币{coin})")
    
    def binomial_prob(self, heads, theta):
        """计算二项分布概率"""
        return comb(self.n_tosses, heads) * (theta ** heads) * ((1 - theta) ** (self.n_tosses - heads))
    
    def joint_probability(self, heads, theta, pi):
        """计算联合概率 p(结果, 硬币) = p(选择硬币) × p(结果|硬币)"""
        return pi * self.binomial_prob(heads, theta)
    
    def marginal_probability(self, heads, theta_A, theta_B, pi_A):
        """计算边际概率 p(结果) = p(结果,硬币A) + p(结果,硬币B)"""
        prob_A = self.joint_probability(heads, theta_A, pi_A)
        prob_B = self.joint_probability(heads, theta_B, 1 - pi_A)
        return prob_A + prob_B
    
    def posterior_probability(self, heads, theta_A, theta_B, pi_A):
        """计算后验概率 p(硬币A|结果) = p(结果,硬币A) / p(结果)"""
        joint_A = self.joint_probability(heads, theta_A, pi_A)
        marginal = self.marginal_probability(heads, theta_A, theta_B, pi_A)
        return joint_A / marginal
    
    def e_step(self, theta_A, theta_B, pi_A):
        """E步：计算每轮使用各硬币的后验概率"""
        gamma_A = []  # 每轮使用硬币A的概率
        gamma_B = []  # 每轮使用硬币B的概率
        
        for heads, tails in self.results:
            post_A = self.posterior_probability(heads, theta_A, theta_B, pi_A)
            gamma_A.append(post_A)
            gamma_B.append(1 - post_A)
        
        return np.array(gamma_A), np.array(gamma_B)
    
    def m_step(self, gamma_A, gamma_B):
        """M步：更新参数"""
        # 计算每轮的正反面次数
        heads_array = np.array([h for h, t in self.results])
        
        # 更新选择硬币A的概率
        pi_A_new = np.mean(gamma_A)
        
        # 更新硬币A的正面概率
        weighted_heads_A = np.sum(gamma_A * heads_array)
        total_tosses_A = np.sum(gamma_A * self.n_tosses)
        theta_A_new = weighted_heads_A / total_tosses_A if total_tosses_A > 0 else 0.5
        
        # 更新硬币B的正面概率
        weighted_heads_B = np.sum(gamma_B * heads_array)
        total_tosses_B = np.sum(gamma_B * self.n_tosses)
        theta_B_new = weighted_heads_B / total_tosses_B if total_tosses_B > 0 else 0.5
        
        return theta_A_new, theta_B_new, pi_A_new
    
    def log_likelihood(self, theta_A, theta_B, pi_A):
        """计算对数似然（边际概率的对数和）"""
        log_lik = 0
        for heads, tails in self.results:
            marginal_prob = self.marginal_probability(heads, theta_A, theta_B, pi_A)
            log_lik += np.log(marginal_prob) if marginal_prob > 0 else -np.inf
        return log_lik
    
    def run_em(self, n_iterations=20, init_theta_A=0.5, init_theta_B=0.5, init_pi_A=0.5):
        """运行EM算法"""
        # 初始化参数
        theta_A = init_theta_A
        theta_B = init_theta_B
        pi_A = init_pi_A
        
        # 记录迭代过程
        history = {
            'theta_A': [theta_A],
            'theta_B': [theta_B],
            'pi_A': [pi_A],
            'log_likelihood': [self.log_likelihood(theta_A, theta_B, pi_A)],
            'gamma_A': []
        }
        
        print(f"\n=== EM算法开始 ===")
        print(f"初始参数: θ_A={theta_A:.3f}, θ_B={theta_B:.3f}, π_A={pi_A:.3f}")
        
        for iteration in range(n_iterations):
            # E步
            gamma_A, gamma_B = self.e_step(theta_A, theta_B, pi_A)
            history['gamma_A'].append(gamma_A.copy())
            
            # M步
            theta_A, theta_B, pi_A = self.m_step(gamma_A, gamma_B)
            
            # 记录历史
            history['theta_A'].append(theta_A)
            history['theta_B'].append(theta_B)
            history['pi_A'].append(pi_A)
            history['log_likelihood'].append(self.log_likelihood(theta_A, theta_B, pi_A))
            
            # 打印进度
            if iteration < 5 or iteration % 5 == 4 or iteration == n_iterations - 1:
                print(f"迭代 {iteration+1}: θ_A={theta_A:.3f}, θ_B={theta_B:.3f}, π_A={pi_A:.3f}, "
                      f"似然={history['log_likelihood'][-1]:.3f}")
        
        return history
    
    def plot_results(self, history):
        """绘制结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 参数收敛图
        axes[0, 0].plot(history['theta_A'], 'b-o', label='θ_A (估计)')
        axes[0, 0].plot(history['theta_B'], 'r-o', label='θ_B (估计)')
        axes[0, 0].axhline(y=self.true_theta_A, color='b', linestyle='--', alpha=0.7, label='θ_A (真实)')
        axes[0, 0].axhline(y=self.true_theta_B, color='r', linestyle='--', alpha=0.7, label='θ_B (真实)')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('硬币正面概率')
        axes[0, 0].set_title('参数收敛过程')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 选择概率收敛图
        axes[0, 1].plot(history['pi_A'], 'g-o', label='π_A (估计)')
        axes[0, 1].axhline(y=self.pi_A, color='g', linestyle='--', alpha=0.7, label='π_A (真实)')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('选择硬币A的概率')
        axes[0, 1].set_title('选择概率收敛过程')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 似然函数变化
        axes[1, 0].plot(history['log_likelihood'], 'o-', color='purple')
        axes[1, 0].set_xlabel('迭代次数')
        axes[1, 0].set_ylabel('对数似然')
        axes[1, 0].set_title('似然函数单调增加 (EM算法保证)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 后验概率热力图
        gamma_matrix = np.array(history['gamma_A'][-5:])  # 最后5次迭代
        im = axes[1, 1].imshow(gamma_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        axes[1, 1].set_xlabel('实验轮次')
        axes[1, 1].set_ylabel('迭代次数 (最后5次)')
        axes[1, 1].set_title('后验概率 γ (使用硬币A的概率)')
        axes[1, 1].set_xticks(range(self.n_rounds))
        axes[1, 1].set_xticklabels([f'第{i+1}轮' for i in range(self.n_rounds)])
        axes[1, 1].set_yticks(range(5))
        axes[1, 1].set_yticklabels([f'迭代{i+16}' for i in range(5)])
        plt.colorbar(im, ax=axes[1, 1])
        
        # 标记真实硬币选择
        for i, coin in enumerate(self.true_coins):
            color = 'white' if gamma_matrix[-1, i] > 0.5 else 'black'
            axes[1, 1].text(i, 4, coin, ha='center', va='center', 
                           fontweight='bold', color=color, fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_probability_calculations(self, theta_A=0.6, theta_B=0.4, pi_A=0.5):
        """演示概率计算过程"""
        print(f"\n=== 概率计算演示 (使用参数: θ_A={theta_A}, θ_B={theta_B}, π_A={pi_A}) ===")
        
        # 以第1轮为例 (7正3反)
        heads, tails = self.results[0]
        true_coin = self.true_coins[0]
        
        print(f"\n第1轮数据: {heads}正{tails}反 (真实使用硬币{true_coin})")
        
        # 计算联合概率
        joint_A = self.joint_probability(heads, theta_A, pi_A)
        joint_B = self.joint_probability(heads, theta_B, 1-pi_A)
        print(f"联合概率 p(结果,硬币A) = {joint_A:.6f}")
        print(f"联合概率 p(结果,硬币B) = {joint_B:.6f}")
        
        # 计算边际概率
        marginal = self.marginal_probability(heads, theta_A, theta_B, pi_A)
        print(f"边际概率 p(结果) = p(结果,硬币A) + p(结果,硬币B) = {marginal:.6f}")
        
        # 计算后验概率
        post_A = self.posterior_probability(heads, theta_A, theta_B, pi_A)
        post_B = 1 - post_A
        print(f"后验概率 p(硬币A|结果) = p(结果,硬币A) / p(结果) = {post_A:.3f}")
        print(f"后验概率 p(硬币B|结果) = {post_B:.3f}")
        
        # 解释物理意义
        if post_A > 0.5:
            guess = "硬币A"
            confidence = post_A
        else:
            guess = "硬币B" 
            confidence = post_B
            
        print(f"\n基于当前模型，我们猜测第1轮使用{guess}的概率为{confidence:.1%}")

# 运行演示
if __name__ == "__main__":
    # 创建EM算法实例
    em = CoinEM(true_theta_A=0.7, true_theta_B=0.3, pi_A=0.6)
    
    # 演示概率计算
    em.demonstrate_probability_calculations(theta_A=0.6, theta_B=0.4, pi_A=0.5)
    
    # 运行EM算法
    history = em.run_em(n_iterations=20, init_theta_A=0.5, init_theta_B=0.5, init_pi_A=0.5)
    
    # 绘制结果
    em.plot_results(history)
    
    # 最终结果对比
    print(f"\n=== 最终结果对比 ===")
    print(f"真实参数: θ_A={em.true_theta_A:.3f}, θ_B={em.true_theta_B:.3f}, π_A={em.pi_A:.3f}")
    print(f"估计参数: θ_A={history['theta_A'][-1]:.3f}, θ_B={history['theta_B'][-1]:.3f}, π_A={history['pi_A'][-1]:.3f}")
    
    print(f"\n=== 隐藏状态推断 ===")
    final_gamma = history['gamma_A'][-1]
    for i, (gamma, true_coin) in enumerate(zip(final_gamma, em.true_coins)):
        inferred_coin = "A" if gamma > 0.5 else "B"
        correct = "✓" if inferred_coin == true_coin else "✗"
        print(f"第{i+1}轮: 推断硬币{inferred_coin} (概率{gamma:.3f}), 真实硬币{true_coin} {correct}")