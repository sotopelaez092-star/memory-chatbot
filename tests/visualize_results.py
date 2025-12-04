"""
可视化测试结果
需要安装：pip install matplotlib
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_results(filename="benchmark_results.json"):
    """加载测试结果"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_compression_rate(results):
    """绘制压缩率对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cases = list(results.keys())
    compressors = ["SlidingWindow", "LLMSummary", "Hybrid", "TokenBased"]
    
    data = {comp: [] for comp in compressors}
    
    for case in cases:
        for result in results[case]:
            comp_name = result['compressor']
            rate = float(result['token_reduction'].replace('%', ''))
            data[comp_name].append(rate)
    
    x = range(len(cases))
    width = 0.2
    
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        ax.bar([p + offset for p in x], data[comp], width, label=comp)
    
    ax.set_xlabel('对话场景')
    ax.set_ylabel('Token压缩率 (%)')
    ax.set_title('不同压缩策略的Token压缩率对比')
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compression_rate.png', dpi=300)
    print("压缩率对比图已保存: compression_rate.png")


def plot_speed(results):
    """绘制速度对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cases = list(results.keys())
    compressors = ["SlidingWindow", "LLMSummary", "Hybrid", "TokenBased"]
    
    data = {comp: [] for comp in compressors}
    
    for case in cases:
        for result in results[case]:
            comp_name = result['compressor']
            time = float(result['time_ms'].replace('ms', ''))
            data[comp_name].append(time)
    
    x = range(len(cases))
    width = 0.2
    
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        ax.bar([p + offset for p in x], data[comp], width, label=comp)
    
    ax.set_xlabel('对话场景')
    ax.set_ylabel('耗时 (ms)')
    ax.set_title('不同压缩策略的速度对比')
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')  # 使用对数刻度，因为差异太大
    
    plt.tight_layout()
    plt.savefig('speed_comparison.png', dpi=300)
    print("速度对比图已保存: speed_comparison.png")


def main():
    """主函数"""
    results = load_results()
    
    print("正在生成可视化图表...")
    plot_compression_rate(results)
    plot_speed(results)
    print("\n图表生成完成！")


if __name__ == "__main__":
    main()