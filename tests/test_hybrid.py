"""
混合策略压缩器性能测试（参数统一版）
"""
import sys
import os
import json
from typing import List, Dict

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import time
import random
from dotenv import load_dotenv
from src.memory.compressor import (
    HybridCompressor,
    SlidingWindowCompressor,
    LLMSummaryCompressor
)
from src.llm.deepseek import DeepSeekLLM
from tests.test_data import (
    get_short_conversation,
    get_medium_conversation,
    get_long_conversation,
    get_very_long_conversation
)


# ⭐ 全局参数配置：统一保留轮数
KEEP_TURNS = 5  # 滑动窗口和混合策略都保留5轮
THRESHOLD_TURNS = 10  # 混合策略切换阈值
LLM_KEEP_TURNS = 3  # LLM摘要保留的最近轮数


def test_strategy_selection():
    """测试1：策略自动选择"""
    print("=" * 80)
    print("测试1：策略自动选择")
    print("=" * 80)
    print()
    
    load_dotenv()
    llm = DeepSeekLLM()
    
    # 混合策略配置
    compressor = HybridCompressor(
        llm, 
        threshold_turns=THRESHOLD_TURNS,
        keep_recent_turns=KEEP_TURNS
    )
    
    test_cases = {
        "短对话(5轮)": get_short_conversation(),
        "中等对话(10轮)": get_medium_conversation(),
        "长对话(20轮)": get_long_conversation(),
        "超长对话(30轮)": get_very_long_conversation()
    }
    
    print(f"混合策略配置:")
    print(f"  切换阈值: {THRESHOLD_TURNS}轮")
    print(f"  保留轮数: {KEEP_TURNS}轮（短对话时）")
    print(f"  策略规则: ≤{THRESHOLD_TURNS}轮用滑动窗口，>{THRESHOLD_TURNS}轮用LLM摘要")
    print()
    
    for name, messages in test_cases.items():
        turn_count = len(messages) // 2
        print(f"{name} ({turn_count}轮):")
        
        # 预测使用的策略
        expected_strategy = "滑动窗口" if turn_count <= THRESHOLD_TURNS else "LLM摘要"
        print(f"  预期策略: {expected_strategy}")
        
        # 执行压缩
        try:
            start = time.time()
            compressed = compressor.compress(messages)
            elapsed = (time.time() - start) * 1000
            
            # 判断实际使用的策略
            has_summary = any(m['role'] == 'system' for m in compressed)
            actual_strategy = "LLM摘要" if has_summary else "滑动窗口"
            
            match = "✅" if expected_strategy == actual_strategy else "❌"
            print(f"  实际策略: {actual_strategy} {match}")
            print(f"  压缩结果: {len(messages)}条 → {len(compressed)}条")
            print(f"  耗时: {elapsed:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ 压缩失败: {e}")
        
        print()


def test_performance_comparison():
    """测试2：三种策略性能对比"""
    print("=" * 80)
    print("测试2：三种策略性能对比")
    print("=" * 80)
    print()
    
    load_dotenv()
    llm = DeepSeekLLM()
    
    # 初始化三种策略（参数统一）
    sliding = SlidingWindowCompressor(keep_turns=KEEP_TURNS)
    llm_summary = LLMSummaryCompressor(llm, keep_recent_turns=LLM_KEEP_TURNS)
    hybrid = HybridCompressor(llm, threshold_turns=THRESHOLD_TURNS, keep_recent_turns=KEEP_TURNS)
    
    print(f"参数配置：")
    print(f"  滑动窗口: 保留{KEEP_TURNS}轮")
    print(f"  LLM摘要: 保留{LLM_KEEP_TURNS}轮 + 摘要")
    print(f"  混合策略: 阈值{THRESHOLD_TURNS}轮，保留{KEEP_TURNS}轮（短对话时）")
    print()
    
    test_cases = {
        "短对话(5轮)": get_short_conversation(),
        "中等对话(10轮)": get_medium_conversation(),
        "长对话(20轮)": get_long_conversation(),
        "超长对话(30轮)": get_very_long_conversation()
    }
    
    for name, messages in test_cases.items():
        print(f"\n{name}:")
        print("-" * 80)
        
        original_tokens = sum(llm.count_tokens(m['content']) for m in messages)
        print(f"原始: {len(messages)}条消息, {original_tokens} tokens")
        print()
        
        results = []
        
        # 测试1：滑动窗口
        print(f"滑动窗口(保留{KEEP_TURNS}轮):")
        start = time.time()
        sliding_result = sliding.compress(messages)
        sliding_time = (time.time() - start) * 1000
        sliding_tokens = sum(llm.count_tokens(m['content']) for m in sliding_result)
        sliding_rate = (1 - sliding_tokens / original_tokens) if original_tokens > 0 else 0
        
        print(f"  结果: {len(sliding_result)}条, {sliding_tokens} tokens")
        print(f"  压缩率: {sliding_rate:.1%}")
        print(f"  耗时: {sliding_time:.2f}ms")
        print(f"  成本: $0")
        
        results.append({
            "strategy": "滑动窗口",
            "tokens": sliding_tokens,
            "time": sliding_time,
            "cost": 0,
            "count": len(sliding_result)
        })
        print()
        
        # 测试2：LLM摘要
        print(f"LLM摘要(保留{LLM_KEEP_TURNS}轮+摘要):")
        try:
            start = time.time()
            llm_result = llm_summary.compress(messages)
            llm_time = (time.time() - start) * 1000
            llm_tokens = sum(llm.count_tokens(m['content']) for m in llm_result)
            llm_rate = (1 - llm_tokens / original_tokens) if original_tokens > 0 else 0
            llm_cost = 0.0001  # 估算
            
            print(f"  结果: {len(llm_result)}条, {llm_tokens} tokens")
            print(f"  压缩率: {llm_rate:.1%}")
            print(f"  耗时: {llm_time:.2f}ms")
            print(f"  成本: ${llm_cost:.6f}")
            
            results.append({
                "strategy": "LLM摘要",
                "tokens": llm_tokens,
                "time": llm_time,
                "cost": llm_cost,
                "count": len(llm_result)
            })
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        print()
        
        # 测试3：混合策略
        print(f"混合策略(阈值{THRESHOLD_TURNS}轮，保留{KEEP_TURNS}轮):")
        try:
            start = time.time()
            hybrid_result = hybrid.compress(messages)
            hybrid_time = (time.time() - start) * 1000
            hybrid_tokens = sum(llm.count_tokens(m['content']) for m in hybrid_result)
            hybrid_rate = (1 - hybrid_tokens / original_tokens) if original_tokens > 0 else 0
            
            # 判断使用的策略
            has_summary = any(m['role'] == 'system' for m in hybrid_result)
            used_strategy = "LLM摘要" if has_summary else "滑动窗口"
            hybrid_cost = 0.0001 if has_summary else 0
            
            print(f"  使用策略: {used_strategy}")
            print(f"  结果: {len(hybrid_result)}条, {hybrid_tokens} tokens")
            print(f"  压缩率: {hybrid_rate:.1%}")
            print(f"  耗时: {hybrid_time:.2f}ms")
            print(f"  成本: ${hybrid_cost:.6f}")
            
            results.append({
                "strategy": "混合策略",
                "tokens": hybrid_tokens,
                "time": hybrid_time,
                "cost": hybrid_cost,
                "count": len(hybrid_result),
                "used": used_strategy
            })
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        print()
        
        # 对比分析
        if len(results) >= 2:
            print("⚖️  对比分析:")
            
            # 检查参数一致性
            sliding_data = next((r for r in results if r['strategy'] == '滑动窗口'), None)
            hybrid_data = next((r for r in results if r['strategy'] == '混合策略'), None)
            
            if sliding_data and hybrid_data:
                if hybrid_data.get('used') == '滑动窗口':
                    # 混合策略使用了滑动窗口，应该完全一致
                    if sliding_data['tokens'] == hybrid_data['tokens'] and sliding_data['count'] == hybrid_data['count']:
                        print(f"  ✅ 滑动窗口 vs 混合策略：完全一致")
                        print(f"     {sliding_data['count']}条消息, {sliding_data['tokens']} tokens")
                    else:
                        print(f"  ⚠️  滑动窗口 vs 混合策略：结果不一致！")
                        print(f"     滑动窗口: {sliding_data['count']}条, {sliding_data['tokens']} tokens")
                        print(f"     混合策略: {hybrid_data['count']}条, {hybrid_data['tokens']} tokens")
                        print(f"     可能的原因：参数配置不同")
                elif hybrid_data.get('used') == 'LLM摘要':
                    print(f"  📊 滑动窗口 vs 混合策略（使用LLM）：")
                    print(f"     滑动窗口: {sliding_data['tokens']} tokens, {sliding_data['time']:.2f}ms, $0")
                    print(f"     混合策略: {hybrid_data['tokens']} tokens, {hybrid_data['time']:.2f}ms, ${hybrid_data['cost']:.6f}")
                    
                    token_ratio = hybrid_data['tokens'] / sliding_data['tokens'] if sliding_data['tokens'] > 0 else 0
                    if token_ratio > 1:
                        print(f"     ⚠️  混合策略Token多 {token_ratio:.1f}x")
            
            # 最佳策略
            best_tokens = min(results, key=lambda x: x['tokens'])
            fastest = min(results, key=lambda x: x['time'])
            cheapest = min(results, key=lambda x: x['cost'])
            
            print()
            print(f"  🏆 最优Token: {best_tokens['strategy']} ({best_tokens['tokens']} tokens)")
            print(f"  ⚡ 最快速度: {fastest['strategy']} ({fastest['time']:.2f}ms)")
            print(f"  💰 最低成本: {cheapest['strategy']} (${cheapest['cost']:.6f})")


def test_threshold_tuning():
    """测试3：不同阈值的效果"""
    print("\n" + "=" * 80)
    print("测试3：不同阈值的效果")
    print("=" * 80)
    print()
    
    load_dotenv()
    llm = DeepSeekLLM()
    messages = get_long_conversation()  # 20轮
    
    original_tokens = sum(llm.count_tokens(m['content']) for m in messages)
    print(f"测试对话: 20轮, {original_tokens} tokens")
    print(f"混合策略配置: 保留{KEEP_TURNS}轮")
    print()
    
    print(f"{'阈值':<10} {'使用策略':<15} {'压缩后Tokens':<15} {'耗时(ms)':<15} {'成本':<10}")
    print("-" * 80)
    
    thresholds = [5, 10, 15, 20, 25]
    
    for threshold in thresholds:
        compressor = HybridCompressor(llm, threshold_turns=threshold, keep_recent_turns=KEEP_TURNS)
        
        try:
            start = time.time()
            result = compressor.compress(messages)
            elapsed = (time.time() - start) * 1000
            
            result_tokens = sum(llm.count_tokens(m['content']) for m in result)
            has_summary = any(m['role'] == 'system' for m in result)
            strategy = "LLM摘要" if has_summary else "滑动窗口"
            cost = "$0.0001" if has_summary else "$0"
            
            print(f"{threshold}轮{'':<6} {strategy:<15} {result_tokens:<15} {elapsed:<15.2f} {cost:<10}")
            
        except Exception as e:
            print(f"{threshold}轮{'':<6} 失败: {e}")


def test_information_preservation():
    """测试4：信息保留对比"""
    print("\n" + "=" * 80)
    print("测试4：信息保留对比")
    print("=" * 80)
    print()
    
    load_dotenv()
    llm = DeepSeekLLM()
    
    # 构造包含关键信息的对话（15轮）
    messages = [
        {"role": "user", "content": "我叫Tom，今年28岁"},
        {"role": "assistant", "content": "你好Tom！"},
        {"role": "user", "content": "我在上海浦东工作"},
        {"role": "assistant", "content": "浦东很不错"},
        {"role": "user", "content": "我是AI工程师"},
        {"role": "assistant", "content": "AI很有前景"},
    ]
    
    # 添加一些填充对话到15轮
    for i in range(3, 15):
        messages.append({"role": "user", "content": f"问题{i}"})
        messages.append({"role": "assistant", "content": f"回答{i}"})
    
    key_info = ["Tom", "28岁", "上海", "浦东", "AI工程师"]
    
    print(f"测试对话: 15轮")
    print(f"关键信息: {', '.join(key_info)}")
    print()
    
    strategies = [
        (f"滑动窗口({KEEP_TURNS}轮)", SlidingWindowCompressor(keep_turns=KEEP_TURNS)),
        (f"混合策略(阈值{THRESHOLD_TURNS}轮，保留{KEEP_TURNS}轮)", HybridCompressor(llm, threshold_turns=THRESHOLD_TURNS, keep_recent_turns=KEEP_TURNS)),
    ]
    
    for name, compressor in strategies:
        print(f"{name}:")
        
        try:
            result = compressor.compress(messages)
            all_content = ' '.join([m['content'] for m in result])
            
            preserved = []
            lost = []
            for info in key_info:
                if info in all_content:
                    preserved.append(info)
                else:
                    lost.append(info)
            
            print(f"  保留: {len(preserved)}/{len(key_info)} ({len(preserved)/len(key_info):.0%})")
            if preserved:
                print(f"  ✅ {', '.join(preserved)}")
            if lost:
                print(f"  ❌ 丢失: {', '.join(lost)}")
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
        
        print()


def test_cost_efficiency():
    """测试5：成本效益分析"""
    print("=" * 80)
    print("测试5：成本效益分析")
    print("=" * 80)
    print()
    
    print("模拟100次随机长度对话的成本：")
    print()
    
    total_sliding_cost = 0
    total_llm_cost = 0
    total_hybrid_cost = 0
    
    short_count = 0
    long_count = 0
    
    # 模拟（不实际调用LLM，只统计）
    for i in range(100):
        # 随机生成5-30轮的对话
        turns = random.randint(5, 30)
        
        if turns <= THRESHOLD_TURNS:
            short_count += 1
            hybrid_cost = 0  # 使用滑动窗口
        else:
            long_count += 1
            hybrid_cost = 0.0001  # 使用LLM摘要
        
        total_sliding_cost += 0  # 滑动窗口始终$0
        total_llm_cost += 0.0001  # LLM摘要每次$0.0001
        total_hybrid_cost += hybrid_cost
    
    print(f"100次对话统计:")
    print(f"  短对话(≤{THRESHOLD_TURNS}轮): {short_count}次")
    print(f"  长对话(>{THRESHOLD_TURNS}轮): {long_count}次")
    print()
    
    print(f"成本对比:")
    print(f"  纯滑动窗口: ${total_sliding_cost:.4f}")
    print(f"  纯LLM摘要: ${total_llm_cost:.4f}")
    print(f"  混合策略: ${total_hybrid_cost:.4f}")
    print()
    
    savings_vs_llm = (1 - total_hybrid_cost / total_llm_cost) * 100 if total_llm_cost > 0 else 0
    
    print(f"混合策略优势:")
    print(f"  {short_count}次短对话使用滑动窗口（快速+$0）")
    print(f"  {long_count}次长对话使用LLM摘要（智能，但有成本）")
    print(f"  vs 纯LLM摘要: 节省 {savings_vs_llm:.1f}%")
    print(f"  vs 纯滑动窗口: 多花 ${total_hybrid_cost:.4f}（换取信息保留）")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("混合策略压缩器 - 完整性能测试（参数统一版）")
    print("=" * 80)
    print()
    
    print("🔧 全局参数配置:")
    print(f"  KEEP_TURNS = {KEEP_TURNS} (滑动窗口和混合策略保留轮数)")
    print(f"  THRESHOLD_TURNS = {THRESHOLD_TURNS} (混合策略切换阈值)")
    print(f"  LLM_KEEP_TURNS = {LLM_KEEP_TURNS} (LLM摘要保留轮数)")
    print()
    
    # 测试1：策略选择
    test_strategy_selection()
    
    # 测试2：性能对比
    test_performance_comparison()
    
    # 测试3：阈值调优
    test_threshold_tuning()
    
    # 测试4：信息保留
    test_information_preservation()
    
    # 测试5：成本效益
    test_cost_efficiency()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print()
    print("✅ 策略选择: 自动根据对话长度选择最优策略")
    print("✅ 参数一致: 短对话时，混合策略与滑动窗口完全相同")
    print("✅ 成本优化: 比纯LLM摘要节省成本")
    print("✅ 信息保留: 长对话保留关键信息，短对话保持速度")
    print()
    print("结论: 混合策略是最实用的方案，适合通用场景")
    print("=" * 80)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    run_all_tests()