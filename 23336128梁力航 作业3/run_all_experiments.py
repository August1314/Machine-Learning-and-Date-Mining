"""
运行所有实验的脚本
依次执行练习一和练习二
"""
import sys
import time

def run_experiment(script_name, description):
    """运行单个实验脚本"""
    print("\n" + "=" * 70)
    print(f"开始运行: {description}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        with open(script_name, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code, {'__name__': '__main__'})
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} 完成！")
        print(f"  运行时间: {elapsed_time:.2f} 秒")
        return True
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {description} 失败！")
        print(f"  错误: {e}")
        print(f"  运行时间: {elapsed_time:.2f} 秒")
        return False


def main():
    print("=" * 70)
    print("机器学习作业 - 完整实验运行")
    print("=" * 70)
    print("\n注意: 练习一需要较长时间（约5-10分钟），请耐心等待...")
    
    input("\n按回车键开始运行所有实验...")
    
    total_start = time.time()
    
    # 运行练习一
    success1 = run_experiment('exercise_one.py', '练习一：线性回归')
    
    # 运行练习二
    success2 = run_experiment('exercise_two.py', '练习二：逻辑回归')
    
    total_elapsed = time.time() - total_start
    
    # 总结
    print("\n" + "=" * 70)
    print("实验运行总结")
    print("=" * 70)
    print(f"练习一（线性回归）: {'✓ 成功' if success1 else '✗ 失败'}")
    print(f"练习二（逻辑回归）: {'✓ 成功' if success2 else '✗ 失败'}")
    print(f"\n总运行时间: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
    
    if success1 and success2:
        print("\n✓ 所有实验成功完成！")
        print("\n生成的图表保存在 analysis/ 目录中:")
        print("  - linear_gd_lr_0.00015.png")
        print("  - linear_gd_lr_0.0002.png")
        print("  - linear_sgd.png")
        print("  - logistic_convergence.png")
        print("  - logistic_training_size_analysis.png")
    else:
        print("\n✗ 部分实验失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main()
