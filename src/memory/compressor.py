"""
记忆压缩策略模块

实现了4种不同的压缩策略：
1. SlidingWindowCompressor - 滑动窗口
2. LLMSummaryCompressor - LLM摘要
3. HybridCompressor - 混合策略
4. TokenBasedCompressor - Token动态
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Callable, Optional
import time


class BaseCompressor(ABC):
    """
    压缩策略抽象基类
    
    所有压缩策略都要继承这个类并实现compress方法
    """
    
    @abstractmethod
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        压缩消息列表
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表
        """
        pass
    
    def get_stats(self, 
                  messages: List[Dict[str, str]], 
                  token_counter: Optional[Callable] = None) -> Dict:
        """
        获取压缩统计信息
        
        Args:
            messages: 原始消息列表
            token_counter: token计数函数（可选）
            
        Returns:
            统计信息字典
        """
        # 原始数据
        original_count = len(messages)
        original_tokens = 0
        if token_counter:
            original_tokens = sum(token_counter(m['content']) for m in messages)
        
        # 压缩
        start_time = time.time()
        compressed = self.compress(messages)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # 压缩后数据
        compressed_count = len(compressed)
        compressed_tokens = 0
        if token_counter:
            compressed_tokens = sum(token_counter(m['content']) for m in compressed)
        
        # 计算压缩率
        message_reduction = (1 - compressed_count / original_count) if original_count > 0 else 0
        token_reduction = (1 - compressed_tokens / original_tokens) if original_tokens > 0 else 0
        
        return {
            "strategy": self.__class__.__name__,
            "original_messages": original_count,
            "compressed_messages": compressed_count,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "message_reduction": f"{message_reduction:.1%}",
            "token_reduction": f"{token_reduction:.1%}",
            "time_ms": f"{elapsed_ms:.2f}",
        }


class SlidingWindowCompressor(BaseCompressor):
    """
    滑动窗口压缩策略
    
    原理:
    - 只保留最近N轮对话
    - 最简单、最快速
    - 但会丢失早期信息
    
    优点:
    - 速度极快（<1ms）
    - 实现简单
    - 无额外成本
    
    缺点:
    - 压缩率固定
    - 丢失早期信息
    - 可能丢失重要内容
    
    适用场景:
    - 对话轮数较少（<10轮）
    - 不需要长期上下文
    - 追求极致速度
    """
    
    def __init__(self, keep_turns: int = 5):
        """
        初始化
        
        Args:
            keep_turns: 保留最近几轮对话（1轮 = user + assistant）
        """
        self.keep_turns = keep_turns
        self.keep_messages = keep_turns * 2  # 每轮包含user和assistant两条消息
    
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        压缩：只保留最近N轮
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表（最近N轮）
        """
        # 如果消息数量没超过限制，直接返回
        if len(messages) <= self.keep_messages:
            return messages
        
        # 只保留最近的N条消息
        return messages[-self.keep_messages:]
    
    def __repr__(self) -> str:
        return f"SlidingWindowCompressor(keep_turns={self.keep_turns})"


class LLMSummaryCompressor(BaseCompressor):
    """
    LLM摘要压缩策略
    
    原理:
    - 保留最近N轮完整对话
    - 历史部分用LLM生成摘要
    - 返回：摘要 + 最近完整对话
    
    优点:
    - 压缩率高（70-90%）
    - 保留语义信息
    - 不丢失关键内容
    
    缺点:
    - 速度慢（1-3秒）
    - 有成本（调用LLM）
    - 可能丢失细节
    
    适用场景:
    - 对话轮数很多（>20轮）
    - 需要长期上下文
    - 不追求实时响应
    """
    
    def __init__(self, llm, keep_recent_turns: int = 3):
        """
        初始化
        
        Args:
            llm: LLM实例（需要有chat方法）
            keep_recent_turns: 保留最近几轮完整对话
        """
        self.llm = llm
        self.keep_recent_turns = keep_recent_turns
        self.keep_recent_messages = keep_recent_turns * 2
    
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        压缩：历史摘要 + 最近完整
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表：[摘要消息, 最近消息1, 最近消息2, ...]
        """
        # 如果消息数量不多，不需要压缩
        if len(messages) <= self.keep_recent_messages:
            return messages
        
        # 分割：历史 + 最近
        recent_messages = messages[-self.keep_recent_messages:]
        history_messages = messages[:-self.keep_recent_messages]
        
        # 如果没有历史消息，直接返回
        if not history_messages:
            return recent_messages
        
        # 用LLM生成历史摘要
        summary = self._summarize(history_messages)
        
        # 组合：摘要 + 最近完整
        compressed = [
            {"role": "system", "content": f"📝 对话历史摘要：\n{summary}"}
        ] + recent_messages
        
        return compressed
    
    def _summarize(self, messages: List[Dict[str, str]]) -> str:
        """
        用LLM生成摘要
        
        Args:
            messages: 要摘要的消息列表
            
        Returns:
            摘要文本
        """
        # 格式化消息为文本
        conversation_text = self._format_messages(messages)
        
        # 构建摘要prompt
        prompt = f"""请总结以下对话的关键信息，要求：

1. 提取用户的基本信息（姓名、地点、职业等）
2. 记录用户的偏好和兴趣
3. 列出讨论的主要话题
4. 保留重要的决定或结论
5. 保留具体的数字、名称等关键细节
6. 按时间顺序组织
7. 简洁准确，不超过300字

对话内容：
{conversation_text}

请用简洁的语言总结上述对话的关键信息："""
        
        try:
            # 调用LLM
            summary_messages = [{"role": "user", "content": prompt}]
            summary = self.llm.chat(summary_messages, max_tokens=500, temperature=0.3)
            return summary.strip()
        except Exception as e:
            # 如果LLM调用失败，返回简单的文本摘要
            return f"[摘要生成失败: {str(e)}]\n历史对话包含 {len(messages)} 条消息"
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        格式化消息为文本
        
        Args:
            messages: 消息列表
            
        Returns:
            格式化的文本
        """
        lines = []
        for msg in messages:
            role = "用户" if msg['role'] == 'user' else "助手"
            content = msg['content']
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"LLMSummaryCompressor(keep_recent_turns={self.keep_recent_turns})"


class HybridCompressor(BaseCompressor):
    """
    混合压缩策略
    
    原理:
    - 对话少时：用滑动窗口（快速）
    - 对话多时：用LLM摘要（高效）
    - 自动选择最佳策略
    
    优点:
    - 平衡速度和效果
    - 自适应
    - 兼顾成本和性能
    
    缺点:
    - 逻辑稍复杂
    
    适用场景:
    - 通用场景（推荐）
    - 生产环境
    - 对话长度不确定
    """
    
    def __init__(self, 
                 llm,
                 threshold_turns: int = 10,
                 keep_recent_turns: int = 3):
        """
        初始化
        
        Args:
            llm: LLM实例
            threshold_turns: 触发摘要的阈值（轮数）
            keep_recent_turns: 保留最近几轮完整对话
        """
        self.threshold_turns = threshold_turns
        self.threshold_messages = threshold_turns * 2
        
        # 两个子压缩器
        self.sliding_window = SlidingWindowCompressor(keep_turns=keep_recent_turns)
        self.llm_summary = LLMSummaryCompressor(llm, keep_recent_turns=keep_recent_turns)
    
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        压缩：根据消息数量自动选择策略
        
        策略选择：
        - 消息数 <= 阈值：使用滑动窗口（快）
        - 消息数 > 阈值：使用LLM摘要（效果好）
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表
        """
        message_count = len(messages)
        
        # 策略选择
        if message_count <= self.threshold_messages:
            # 对话不多：用滑动窗口（快）
            return self.sliding_window.compress(messages)
        else:
            # 对话很多：用LLM摘要（效果好）
            return self.llm_summary.compress(messages)
    
    def get_current_strategy(self, messages: List[Dict[str, str]]) -> str:
        """
        获取当前会使用的策略
        
        Args:
            messages: 消息列表
            
        Returns:
            策略名称
        """
        if len(messages) <= self.threshold_messages:
            return "SlidingWindow"
        else:
            return "LLMSummary"
    
    def __repr__(self) -> str:
        return f"HybridCompressor(threshold_turns={self.threshold_turns})"


class TokenBasedCompressor(BaseCompressor):
    """
    基于Token的动态压缩策略
    
    原理:
    - 不按轮数，按token预算
    - 从最早的消息开始删除，直到满足预算
    - 精确控制token数量
    
    优点:
    - 精确控制token数量
    - 充分利用上下文窗口
    - 适应不同长度的消息
    
    缺点:
    - 可能切断对话
    - 需要token计数器
    
    适用场景:
    - 有严格token限制
    - 需要精确控制成本
    - 消息长度差异大
    """
    
    def __init__(self, token_counter: Callable, max_tokens: int = 4000):
        """
        初始化
        
        Args:
            token_counter: token计数函数
            max_tokens: 最大token数
        """
        self.token_counter = token_counter
        self.max_tokens = max_tokens
    
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        压缩：保持在token预算内
        
        策略：
        1. 计算当前总token数
        2. 如果超预算，从最早的消息开始删除
        3. 直到满足预算
        
        Args:
            messages: 原始消息列表
            
        Returns:
            压缩后的消息列表
        """
        if not messages:
            return messages
        
        # 计算当前总token数
        total_tokens = sum(self.token_counter(m['content']) for m in messages)
        
        # 如果没超预算，直接返回
        if total_tokens <= self.max_tokens:
            return messages
        
        # 从最早的消息开始删除，直到满足预算
        compressed = messages[:]
        current_tokens = total_tokens
        
        while compressed and current_tokens > self.max_tokens:
            # 删除最早的消息
            removed = compressed.pop(0)
            removed_tokens = self.token_counter(removed['content'])
            current_tokens -= removed_tokens
        
        return compressed
    
    def get_token_count(self, messages: List[Dict[str, str]]) -> int:
        """
        获取消息列表的总token数
        
        Args:
            messages: 消息列表
            
        Returns:
            总token数
        """
        return sum(self.token_counter(m['content']) for m in messages)
    
    def __repr__(self) -> str:
        return f"TokenBasedCompressor(max_tokens={self.max_tokens})"


# ==================== 测试代码 ====================

def create_test_messages(num_turns: int = 10) -> List[Dict[str, str]]:
    """创建测试消息"""
    messages = []
    
    # 模拟真实对话
    conversations = [
        ("我叫Tom，在上海工作", "你好Tom！很高兴认识你。在上海工作一定很忙吧？"),
        ("是的，我是一名软件工程师", "软件工程师是个很好的职业！你主要做什么方向的开发？"),
        ("我主要做AI方向，最近在研究Agent", "AI Agent是个很热门的方向！有什么具体的研究内容吗？"),
        ("我在研究多Agent协作", "多Agent协作确实很有挑战性。有遇到什么技术难题吗？"),
        ("主要是Agent之间的通信和协调", "这确实是个核心问题。你考虑过用消息队列吗？"),
        ("考虑过，但觉得可能太重了", "确实，可以先从简单的方案开始。比如共享内存？"),
        ("对，我现在用的就是共享内存", "很好的选择！性能如何？"),
        ("还不错，QPS能到1000+", "很优秀的性能！有做性能优化吗？"),
        ("做了一些缓存优化", "缓存是个好方法。用的什么缓存策略？"),
        ("LRU缓存，效果还可以", "LRU是经典方案。考虑过其他策略吗？"),
    ]
    
    for i in range(min(num_turns, len(conversations))):
        user_msg, assistant_msg = conversations[i]
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # 如果需要更多轮，用简单消息填充
    if num_turns > len(conversations):
        for i in range(len(conversations), num_turns):
            messages.append({"role": "user", "content": f"用户消息 {i+1}"})
            messages.append({"role": "assistant", "content": f"助手回复 {i+1}"})
    
    return messages


def test_sliding_window():
    """测试滑动窗口压缩"""
    print("=" * 60)
    print("测试 1: 滑动窗口压缩")
    print("=" * 60)
    
    messages = create_test_messages(num_turns=10)
    print(f"\n原始消息: {len(messages)}条 (10轮对话)")
    
    compressor = SlidingWindowCompressor(keep_turns=3)
    compressed = compressor.compress(messages)
    
    print(f"压缩后: {len(compressed)}条 (3轮对话)")
    print(f"压缩率: {(1 - len(compressed)/len(messages)):.1%}")
    print("\n保留的消息:")
    for msg in compressed:
        role = "用户" if msg['role'] == 'user' else "助手"
        print(f"  {role}: {msg['content'][:50]}...")
    print()


def test_llm_summary():
    """测试LLM摘要压缩"""
    print("=" * 60)
    print("测试 2: LLM摘要压缩")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        from src.llm.deepseek import DeepSeekLLM
        
        load_dotenv()
        llm = DeepSeekLLM()
        
        messages = create_test_messages(num_turns=10)
        print(f"\n原始消息: {len(messages)}条 (10轮对话)")
        
        compressor = LLMSummaryCompressor(llm, keep_recent_turns=2)
        print("\n正在生成摘要...")
        compressed = compressor.compress(messages)
        
        print(f"压缩后: {len(compressed)}条")
        print("\n压缩后内容:")
        for msg in compressed:
            if msg['role'] == 'system':
                print(f"  [摘要]:\n{msg['content']}\n")
            else:
                role = "用户" if msg['role'] == 'user' else "助手"
                print(f"  {role}: {msg['content']}")
        
        # 统计信息
        stats = compressor.get_stats(messages, llm.count_tokens)
        print(f"\n统计信息:")
        print(f"  消息压缩率: {stats['message_reduction']}")
        print(f"  Token压缩率: {stats['token_reduction']}")
        print(f"  耗时: {stats['time_ms']}ms")
        print()
        
    except Exception as e:
        print(f"错误: {e}")
        print("需要配置DEEPSEEK_API_KEY才能测试LLM摘要\n")


def test_hybrid():
    """测试混合压缩"""
    print("=" * 60)
    print("测试 3: 混合压缩策略")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        from src.llm.deepseek import DeepSeekLLM
        
        load_dotenv()
        llm = DeepSeekLLM()
        
        compressor = HybridCompressor(llm, threshold_turns=5, keep_recent_turns=2)
        
        # 测试短对话
        print("\n场景1: 短对话（4轮）")
        messages_short = create_test_messages(num_turns=4)
        print(f"原始: {len(messages_short)}条")
        print(f"策略: {compressor.get_current_strategy(messages_short)}")
        compressed = compressor.compress(messages_short)
        print(f"压缩后: {len(compressed)}条\n")
        
        # 测试长对话
        print("场景2: 长对话（10轮）")
        messages_long = create_test_messages(num_turns=10)
        print(f"原始: {len(messages_long)}条")
        print(f"策略: {compressor.get_current_strategy(messages_long)}")
        print("正在压缩...")
        compressed = compressor.compress(messages_long)
        print(f"压缩后: {len(compressed)}条\n")
        
    except Exception as e:
        print(f"错误: {e}")
        print("需要配置DEEPSEEK_API_KEY才能测试混合策略\n")


def test_token_based():
    """测试Token-based压缩"""
    print("=" * 60)
    print("测试 4: Token-based动态压缩")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        from src.llm.deepseek import DeepSeekLLM
        
        load_dotenv()
        llm = DeepSeekLLM()
        
        messages = create_test_messages(num_turns=10)
        original_tokens = sum(llm.count_tokens(m['content']) for m in messages)
        
        print(f"\n原始: {len(messages)}条消息, {original_tokens} tokens")
        
        # 测试不同的token预算
        budgets = [500, 300, 200]
        compressor = TokenBasedCompressor(llm.count_tokens, max_tokens=500)
        
        for budget in budgets:
            compressor.max_tokens = budget
            compressed = compressor.compress(messages)
            compressed_tokens = compressor.get_token_count(compressed)
            
            print(f"\nToken预算: {budget}")
            print(f"  压缩后: {len(compressed)}条消息, {compressed_tokens} tokens")
            print(f"  压缩率: {(1 - compressed_tokens/original_tokens):.1%}")
        print()
        
    except Exception as e:
        print(f"错误: {e}")
        print("需要配置DEEPSEEK_API_KEY才能测试Token-based压缩\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("记忆压缩策略测试")
    print("=" * 60 + "\n")
    
    # 测试所有策略
    test_sliding_window()
    test_llm_summary()
    test_hybrid()
    test_token_based()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)