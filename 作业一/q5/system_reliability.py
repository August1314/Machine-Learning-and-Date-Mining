import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SystemReliability:
    def __init__(self):
        """
        初始化系统可靠性仿真
        
        系统结构（并联）：
        - 上支路：组件A（可靠性85%）
        - 下支路：组件B与组件C串联（可靠性 0.95×0.90=0.855）
        - 系统成功条件：A 成功，或 B 与 C 同时成功
        """
        self.component_reliability = {
            'A': 0.85,
            'B': 0.95,
            'C': 0.90
        }
        # 并联系统不需要路径选择概率
        
        # 计算理论可靠性
        self.theoretical_reliability = self.calculate_theoretical_reliability()
        
    def calculate_theoretical_reliability(self):
        """
        计算理论系统可靠性
        
        返回:
        float: 理论可靠性
        """
        # 并联系统：R = 1 - (1-RA)*(1-RB*RC)
        reliability_bc = self.component_reliability['B'] * self.component_reliability['C']
        reliability = 1.0 - (1.0 - self.component_reliability['A']) * (1.0 - reliability_bc)
        return reliability
    
    def simulate_single_trial(self):
        """
        仿真单次试验
        
        返回:
        dict: 包含试验结果的字典
        """
        # 同时抽样三个组件是否成功
        r_a = np.random.random()
        r_b = np.random.random()
        r_c = np.random.random()

        success_a = r_a <= self.component_reliability['A']
        success_b = r_b <= self.component_reliability['B']
        success_c = r_c <= self.component_reliability['C']

        success_bc = success_b and success_c
        success = success_a or success_bc

        return {
            'success': success,
            'success_a': success_a,
            'success_b': success_b,
            'success_c': success_c,
            'components': ['A', 'B', 'C']
        }
    
    def monte_carlo_simulation(self, num_trials=10000):
        """
        蒙特卡洛仿真
        
        参数:
        num_trials: 试验次数
        
        返回:
        dict: 包含仿真结果的字典
        """
        print(f"正在进行 {num_trials} 次蒙特卡洛仿真...")
        
        results = []
        successful_trials = 0
        # 统计单个组件的成功次数
        component_successes = {'A': 0, 'B': 0, 'C': 0}
        
        for trial in range(num_trials):
            if trial % 1000 == 0 and trial > 0:
                print(f"已完成 {trial} 次试验...")
            
            result = self.simulate_single_trial()
            results.append(result)
            
            if result['success']:
                successful_trials += 1
            
            # 统计组件成功情况
            component_successes['A'] += 1 if result['success_a'] else 0
            component_successes['B'] += 1 if result['success_b'] else 0
            component_successes['C'] += 1 if result['success_c'] else 0
        
        # 计算统计结果
        empirical_reliability = successful_trials / num_trials
        
        # 组件边际成功率（用于参考）
        component_success_rates = {k: v / num_trials for k, v in component_successes.items()}
        
        print(f"仿真完成！成功次数: {successful_trials}/{num_trials}")
        print(f"经验可靠性: {empirical_reliability:.6f}")
        print(f"理论可靠性: {self.theoretical_reliability:.6f}")
        print(f"误差: {abs(empirical_reliability - self.theoretical_reliability):.6f}")
        
        return {
            'num_trials': num_trials,
            'successful_trials': successful_trials,
            'empirical_reliability': empirical_reliability,
            'theoretical_reliability': self.theoretical_reliability,
            'error': abs(empirical_reliability - self.theoretical_reliability),
            'component_successes': component_successes,
            'component_success_rates': component_success_rates,
            'results': results
        }
    
    def analyze_convergence(self, max_trials=50000, step_size=1000):
        """
        分析收敛性
        
        参数:
        max_trials: 最大试验次数
        step_size: 步长
        
        返回:
        dict: 收敛分析结果
        """
        print(f"正在分析收敛性（最大{max_trials}次试验）...")
        
        trial_counts = range(step_size, max_trials + 1, step_size)
        reliabilities = []
        errors = []
        
        for num_trials in trial_counts:
            sim_result = self.monte_carlo_simulation(num_trials)
            reliabilities.append(sim_result['empirical_reliability'])
            errors.append(sim_result['error'])
        
        return {
            'trial_counts': list(trial_counts),
            'reliabilities': reliabilities,
            'errors': errors
        }
    
    def visualize_results(self, sim_result, convergence_result=None, save_path=None):
        """
        可视化仿真结果
        
        参数:
        sim_result: 仿真结果
        convergence_result: 收敛分析结果
        save_path: 图片保存路径
        """
        if convergence_result:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：组件成功次数（边际）
        paths = list(sim_result['component_successes'].keys())
        counts = [sim_result['num_trials']] * len(paths)
        success_counts = list(sim_result['component_successes'].values())
        
        x = np.arange(len(paths))
        width = 0.35
        
        ax1.bar(x - width/2, counts, width, label='总次数', alpha=0.7, color='skyblue')
        ax1.bar(x + width/2, success_counts, width, label='成功次数', alpha=0.7, color='lightcoral')
        
        ax1.set_xlabel('路径')
        ax1.set_ylabel('次数')
        ax1.set_title('组件成功次数（边际统计）')
        ax1.set_xticks(x)
        ax1.set_xticklabels(paths)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, (count, success) in enumerate(zip(counts, success_counts)):
            ax1.text(i - width/2, count + max(counts)*0.01, str(count), ha='center', va='bottom')
            ax1.text(i + width/2, success + max(counts)*0.01, str(success), ha='center', va='bottom')
        
        # 右图：可靠性对比
        categories = ['理论值(并联)', '经验值']
        values = [sim_result['theoretical_reliability'], sim_result['empirical_reliability']]
        colors = ['lightblue', 'lightgreen']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7)
        ax2.set_ylabel('可靠性')
        ax2.set_title('理论可靠性 vs 经验可靠性')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 添加误差线
        error = sim_result['error']
        ax2.errorbar(1, sim_result['empirical_reliability'], yerr=error, 
                    fmt='r', capsize=5, capthick=2, label=f'误差: {error:.4f}')
        ax2.legend()
        
        if convergence_result:
            # 左下图：收敛性分析
            ax3.plot(convergence_result['trial_counts'], convergence_result['reliabilities'], 
                    'b-', linewidth=2, label='经验可靠性')
            ax3.axhline(y=sim_result['theoretical_reliability'], color='r', 
                       linestyle='--', linewidth=2, label='理论可靠性')
            ax3.set_xlabel('试验次数')
            ax3.set_ylabel('可靠性')
            ax3.set_title('可靠性收敛分析')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 右下图：误差收敛
            ax4.plot(convergence_result['trial_counts'], convergence_result['errors'], 
                    'g-', linewidth=2, label='绝对误差')
            ax4.set_xlabel('试验次数')
            ax4.set_ylabel('绝对误差')
            ax4.set_title('误差收敛分析')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        plt.show()
    
    def detailed_analysis(self, sim_result):
        """
        详细分析仿真结果
        
        参数:
        sim_result: 仿真结果
        """
        print("\n详细分析:")
        print("=" * 50)
        
        # 基本统计
        print(f"总试验次数: {sim_result['num_trials']}")
        print(f"成功次数: {sim_result['successful_trials']}")
        print(f"失败次数: {sim_result['num_trials'] - sim_result['successful_trials']}")
        print(f"经验可靠性: {sim_result['empirical_reliability']:.6f}")
        print(f"理论可靠性: {sim_result['theoretical_reliability']:.6f}")
        print(f"绝对误差: {sim_result['error']:.6f}")
        print(f"相对误差: {sim_result['error'] / sim_result['theoretical_reliability'] * 100:.4f}%")
        
        # 路径分析
        print(f"\n组件边际成功率:")
        print("-" * 30)
        for comp in ['A', 'B', 'C']:
            rate = sim_result['component_success_rates'][comp]
            print(f"组件{comp}: {rate:.4f}")
        
        # 组件分析
        print(f"\n组件可靠性分析:")
        print("-" * 30)
        print(f"组件A: {self.component_reliability['A']:.2f}")
        print(f"组件B: {self.component_reliability['B']:.2f}")
        print(f"组件C: {self.component_reliability['C']:.2f}")
        
        # 理论计算验证
        print(f"\n理论计算验证:")
        print("-" * 30)
        reliability_bc = self.component_reliability['B'] * self.component_reliability['C']
        total_reliability = 1.0 - (1.0 - self.component_reliability['A']) * (1.0 - reliability_bc)
        print(f"R_BC = {self.component_reliability['B']} × {self.component_reliability['C']} = {reliability_bc:.4f}")
        print(f"R_sys = 1 - (1-RA)(1-RBC) = {total_reliability:.5f}")

def main():
    """
    主函数：执行系统可靠性仿真
    """
    print("系统可靠性仿真 - 蒙特卡洛方法")
    print("=" * 50)
    print("系统结构：并联")
    print("- 上支路：A，可靠性0.85")
    print("- 下支路：B与C串联，可靠性0.95×0.90=0.855")
    print("=" * 50)
    
    # 创建系统可靠性实例
    system = SystemReliability()
    
    # 进行蒙特卡洛仿真
    sim_result = system.monte_carlo_simulation(num_trials=10000)
    
    # 详细分析
    system.detailed_analysis(sim_result)
    
    # 收敛性分析
    print("\n正在进行收敛性分析...")
    convergence_result = system.analyze_convergence(max_trials=20000, step_size=2000)
    
    # 可视化结果
    print("\n正在生成可视化图片...")
    system.visualize_results(sim_result, convergence_result, 
                           save_path='/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q5/system_reliability_visualization.png')
    
    # 保存结果
    results_data = {
        'Component_A_Reliability': system.component_reliability['A'],
        'Component_B_Reliability': system.component_reliability['B'],
        'Component_C_Reliability': system.component_reliability['C'],
        'Theoretical_Reliability': sim_result['theoretical_reliability'],
        'Empirical_Reliability': sim_result['empirical_reliability'],
        'Error': sim_result['error'],
        'Relative_Error_Percent': sim_result['error'] / sim_result['theoretical_reliability'] * 100,
        'Total_Trials': sim_result['num_trials'],
        'Successful_Trials': sim_result['successful_trials']
    }
    
    df = pd.DataFrame([results_data])
    df.to_csv('/Users/lianglihang/Downloads/Machine-Learning-and-Date-Mining/作业一/q5/system_reliability_results.csv', 
              index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到 system_reliability_results.csv")
    
    return sim_result, convergence_result

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 运行主程序
    sim_result, convergence_result = main()

