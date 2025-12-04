"""
压缩策略性能测试
对比4种压缩策略的性能表现
"""

import sys
import os
import time
import json
from typing import List, Dict

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dotenv import load_dotenv

from src.llm.deepseek import DeepSeekLLM
from src.memory.compressor import (
    SlidingWindowCompressor,
    LLMSummaryCompressor,
    HybridCompressor,
    TokenBasedCompressor
)
from tests.test_data import (
    get_short_conversation,
    get_medium_conversation,
    get_long_conversation,
    get_very_long_conversation
)


class CompressorBenchmark:
    """压缩器性能测试"""
    
    def __init__(self, llm):
        self.llm = llm
        
        # 初始化4种压缩器
        self.compressors = {
            "SlidingWindow": SlidingWindowCompressor(keep_turns=5),
            "LLMSummary": LLMSummaryCompressor(llm, keep_recent_turns=3),
            "Hybrid": HybridCompressor(llm, threshold_turns=10, keep_recent_turns=3),
            "TokenBased": TokenBasedCompressor(llm.count_tokens, max_tokens=1000)
        }
        
        # 测试数据集
        self.test_cases = {
            "短对话(5轮)": get_short_conversation(),
            "中等对话(10轮)": get_medium_conversation(),
            "长对话(20轮)": get_long_conversation(),
            "超长对话(30轮)": get_very_long_conversation()
        }
    
    def test_single_compressor(self, 
                               name: str, 
                               compressor, 
                               messages: List[Dict],
                               repeat: int = 1) -> Dict:
        """
        测试单个压缩器
        
        Args:
            name: 压缩器名称
            compressor: 压缩器实例
            messages: 测试消息
            repeat: 重复次数（用于测试LLM摘要的稳定性）
            
        Returns:
            性能指标字典
        """
        # 原始数据
        original_count = len(messages)
        original_tokens = sum(self.llm.count_tokens(m['content']) for m in messages)
        
        # 执行压缩（多次测试取平均）
        total_time = 0
        compressed_results = []
        
        for _ in range(repeat):
            start = time.time()
            compressed = compressor.compress(messages)
            elapsed = time.time() - start
            total_time += elapsed
            compressed_results.append(compressed)
        
        avg_time = total_time / repeat
        
        # 使用最后一次的压缩结果计算指标
        compressed = compressed_results[-1]
        compressed_count = len(compressed)
        compressed_tokens = sum(self.llm.count_tokens(m['content']) for m in compressed)
        
        # 计算压缩率
        message_reduction = (1 - compressed_count / original_count) if original_count > 0 else 0
        token_reduction = (1 - compressed_tokens / original_tokens) if original_tokens > 0 else 0
        
        # 估算成本（DeepSeek: $0.14/1M input tokens）
        cost = (original_tokens / 1_000_000) * 0.14 if name == "LLMSummary" else 0
        
        return {
            "compressor": name,
            "original_messages": original_count,
            "compressed_messages": compressed_count,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "message_reduction": f"{message_reduction:.1%}",
            "token_reduction": f"{token_reduction:.1%}",
            "time_ms": f"{avg_time * 1000:.2f}",
            "cost_usd": f"${cost:.6f}"
        }
    
    def run_benchmark(self, repeat: int = 1) -> Dict:
        """
        运行完整的性能测试
        
        Args:
            repeat: 每个测试重复次数
            
        Returns:
            完整的测试结果
        """
        results = {}
        
        print("=" * 80)
        print("压缩策略性能测试")
        print("=" * 80)
        print(f"重复次数: {repeat}")
        print()
        
        for case_name, messages in self.test_cases.items():
            print(f"\n{'='*80}")
            print(f"测试场景: {case_name}")
            print(f"{'='*80}")
            
            case_results = []
            
            for comp_name, compressor in self.compressors.items():
                print(f"\n测试 {comp_name}...", end=" ")
                
                try:
                    result = self.test_single_compressor(
                        comp_name, 
                        compressor, 
                        messages,
                        repeat=repeat if comp_name == "LLMSummary" else 1
                    )
                    case_results.append(result)
                    print("✓")
                except Exception as e:
                    print(f"✗ 错误: {e}")
                    continue
            
            results[case_name] = case_results
            
            # 打印本场景的结果
            self._print_case_results(case_name, case_results)
        
        return results
    
    def _print_case_results(self, case_name: str, results: List[Dict]):
        """打印单个场景的结果"""
        print(f"\n{case_name} - 测试结果:")
        print("-" * 80)
        
        # 表头
        print(f"{'策略':<15} {'消息数':<12} {'Token数':<15} {'压缩率':<10} {'耗时':<12} {'成本':<12}")
        print("-" * 80)
        
        # 数据行
        for r in results:
            msg_str = f"{r['compressed_messages']}/{r['original_messages']}"
            token_str = f"{r['compressed_tokens']}/{r['original_tokens']}"
            
            print(f"{r['compressor']:<15} {msg_str:<12} {token_str:<15} "
                  f"{r['token_reduction']:<10} {r['time_ms']:<12} {r['cost_usd']:<12}")
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """保存结果到文件"""
        filepath = os.path.join(project_root, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {filepath}")
    
    def print_summary(self, results: Dict):
        """打印总结"""
        print("\n" + "=" * 80)
        print("测试总结")
        print("=" * 80)
        
        print("\n📊 速度排名（越快越好）:")
        all_speeds = []
        for case_name, case_results in results.items():
            for r in case_results:
                time_val = float(r['time_ms'].replace('ms', ''))
                all_speeds.append((f"{r['compressor']}", time_val, case_name))
        
        all_speeds.sort(key=lambda x: x[1])
        for i, (name, speed, case) in enumerate(all_speeds[:8], 1):
            print(f"  {i}. {name:<15} {speed:.2f}ms ({case})")
        
        print("\n📉 压缩率排名（越高越好）:")
        all_compressions = []
        for case_name, case_results in results.items():
            if "长对话" in case_name:  # 只看长对话的压缩效果
                for r in case_results:
                    rate = float(r['token_reduction'].replace('%', ''))
                    all_compressions.append((r['compressor'], rate, case_name))
        
        all_compressions.sort(key=lambda x: x[1], reverse=True)
        for i, (name, rate, case) in enumerate(all_compressions, 1):
            print(f"  {i}. {name:<15} {rate:.1f}% ({case})")
        
        print("\n💰 成本对比:")
        print("  SlidingWindow: $0 (无成本)")
        print("  TokenBased:    $0 (无成本)")
        print("  Hybrid:        $0-$0.0001 (根据对话长度)")
        print("  LLMSummary:    $0.0001-$0.0003 (每次压缩)")
        
        print("\n🎯 推荐使用场景:")
        print("  SlidingWindow  → 短对话(<10轮) + 追求极致速度")
        print("  TokenBased     → 严格token限制 + 追求速度")
        print("  Hybrid         → 通用场景 (推荐) ⭐")
        print("  LLMSummary     → 超长对话(>30轮) + 需要保留语义")


def main():
    """主函数"""
    # 加载环境变量
    load_dotenv(os.path.join(project_root, '.env'))
    
    print("\n正在初始化...")
    try:
        llm = DeepSeekLLM()
        print("✓ LLM初始化成功")
    except Exception as e:
        print(f"✗ LLM初始化失败: {e}")
        print("\n请确保：")
        print("  1. .env文件存在")
        print("  2. DEEPSEEK_API_KEY已配置")
        return
    
    benchmark = CompressorBenchmark(llm)
    
    # 运行测试
    print("\n开始测试...")
    results = benchmark.run_benchmark(repeat=1)
    
    # 打印总结
    benchmark.print_summary(results)
    
    # 保存结果
    benchmark.save_results(results)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()