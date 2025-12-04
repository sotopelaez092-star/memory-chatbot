"""
List vs Deque 性能对比测试

测试场景：
1. 连续添加消息
2. 满容量后继续添加
3. 获取所有消息
4. 混合操作
"""

import time
from collections import deque
from typing import List, Dict


class MemoryWithList:
    """使用List实现的记忆"""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        
        # 手动删除超出的消息
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict]:
        return self.messages.copy()


class MemoryWithDeque:
    """使用Deque实现的记忆"""
    
    def __init__(self, max_messages: int = 20):
        self.messages = deque(maxlen=max_messages)
    
    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
    
    def get_messages(self) -> List[Dict]:
        return list(self.messages)


def test_continuous_add(memory_class, n: int = 1000) -> float:
    """
    测试1：连续添加消息
    
    Args:
        memory_class: 内存类
        n: 添加次数
    
    Returns:
        耗时（毫秒）
    """
    memory = memory_class(max_messages=20)
    
    start = time.time()
    for i in range(n):
        memory.add_message("user", f"消息 {i}")
    elapsed = (time.time() - start) * 1000
    
    return elapsed


def test_add_at_full_capacity(memory_class, n: int = 1000) -> float:
    """
    测试2：满容量后继续添加
    
    场景：已经有20条消息，继续添加n条
    这是最常见的场景（对话超过限制后）
    
    Returns:
        耗时（毫秒）
    """
    memory = memory_class(max_messages=20)
    
    # 先填满
    for i in range(20):
        memory.add_message("user", f"初始消息 {i}")
    
    # 测试满容量后的添加
    start = time.time()
    for i in range(n):
        memory.add_message("user", f"新消息 {i}")
    elapsed = (time.time() - start) * 1000
    
    return elapsed


def test_get_messages(memory_class, n: int = 1000) -> float:
    """
    测试3：获取消息
    
    Returns:
        耗时（毫秒）
    """
    memory = memory_class(max_messages=20)
    
    # 先添加20条消息
    for i in range(20):
        memory.add_message("user", f"消息 {i}")
    
    # 测试获取消息的速度
    start = time.time()
    for _ in range(n):
        messages = memory.get_messages()
    elapsed = (time.time() - start) * 1000
    
    return elapsed


def test_mixed_operations(memory_class, n: int = 1000) -> float:
    """
    测试4：混合操作
    
    模拟真实对话场景：
    - 添加用户消息
    - 获取历史
    - 添加AI回复
    - 再次获取历史
    
    Returns:
        耗时（毫秒）
    """
    memory = memory_class(max_messages=20)
    
    start = time.time()
    for i in range(n):
        # 用户输入
        memory.add_message("user", f"用户问题 {i}")
        # 构建上下文（需要获取历史）
        history = memory.get_messages()
        # AI回复
        memory.add_message("assistant", f"AI回答 {i}")
    elapsed = (time.time() - start) * 1000
    
    return elapsed


def run_all_tests():
    """运行所有测试 - 增加测试规模"""
    
    print("=" * 80)
    print("List vs Deque 性能对比测试（大规模）")
    print("=" * 80)
    print()
    
    # 增加测试规模
    test_configs = [
        ("连续添加10000条消息", test_continuous_add, 10000),
        ("满容量后添加10000条", test_add_at_full_capacity, 10000),
        ("获取消息10000次", test_get_messages, 10000),
        ("混合操作10000次", test_mixed_operations, 10000),
        
        # 添加超大规模测试
        ("连续添加100000条消息", test_continuous_add, 100000),
        ("满容量后添加100000条", test_add_at_full_capacity, 100000),
    ]
    
    results = []
    
    for test_name, test_func, n in test_configs:
        print(f"测试场景: {test_name}")
        print("-" * 80)
        
        # 测试List
        print(f"  测试List...", end=" ", flush=True)
        list_time = test_func(MemoryWithList, n)
        print(f"✓ {list_time:.2f}ms")
        
        # 测试Deque
        print(f"  测试Deque...", end=" ", flush=True)
        deque_time = test_func(MemoryWithDeque, n)
        print(f"✓ {deque_time:.2f}ms")
        
        # 计算倍数
        speedup = list_time / deque_time if deque_time > 0 else 0
        winner = "Deque" if speedup > 1 else "List"
        print(f"  → {winner}快 {abs(speedup):.1f}x")
        print()
        
        results.append({
            "test": test_name,
            "list": list_time,
            "deque": deque_time,
            "speedup": speedup
        })
    
    # 打印总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    print()
    print(f"{'测试场景':<30} {'List(ms)':<12} {'Deque(ms)':<12} {'速度对比':<10}")
    print("-" * 80)
    
    for r in results:
        winner = "🟢 Deque" if r['speedup'] > 1 else "🔴 List"
        print(f"{r['test']:<30} {r['list']:<12.2f} {r['deque']:<12.2f} {winner} {abs(r['speedup']):.1f}x")
    
    print()
    print("=" * 80)
    print("关键结论:")
    
    # 找出最大差异
    max_speedup = max(results, key=lambda x: abs(x['speedup'] - 1))
    print(f"  最大差异: {max_speedup['test']}")
    print(f"  → Deque快 {max_speedup['speedup']:.1f}x")
    
    # 计算在添加场景下的平均提升
    add_tests = [r for r in results if '添加' in r['test']]
    avg_add_speedup = sum(r['speedup'] for r in add_tests) / len(add_tests)
    print(f"  添加操作平均: Deque快 {avg_add_speedup:.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()