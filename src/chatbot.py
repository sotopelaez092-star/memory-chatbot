from typing import List, Dict, Optional
from llm.base import BaseLLM
from memory.short_term import ShortTermMemory
from memory.compressor import (
    BaseCompressor,
    SlidingWindowCompressor,
    LLMSummaryCompressor,
    HybridCompressor,
    TokenBasedCompressor
)

class SimpleChatbot:
    """简单的聊天机器人（无记忆版本）"""
    
    def __init__(self, llm: BaseLLM, system_prompt: Optional[str] = None):
        """
        初始化
        
        Args:
            llm: LLM实例
            system_prompt: 系统提示词
        """
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个友好的AI助手。"
    
    def chat(self, user_input: str) -> str:
        """
        单轮对话（无历史记录）
        
        Args:
            user_input: 用户输入
            
        Returns:
            机器人回复
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self.llm.chat(messages)
        return response

class MemoryChatbot:
    """
    带记忆的聊天机器人
    
    特点:
    - 使用短期记忆保存对话历史
    - 自动管理上下文
    """
    
    def __init__(self, 
                 llm: BaseLLM,
                 system_prompt: Optional[str] = None,
                 max_turns: int = 10):
        """
        初始化
        
        Args:
            llm: LLM实例
            system_prompt: 系统提示词
            max_turns: 最大保存轮数
        """
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个友好、乐于助人的AI助手。"
        
        # 初始化短期记忆
        self.memory = ShortTermMemory(max_turns=max_turns)
    
    def chat(self, user_input: str) -> str:
        """
        对话（带记忆）
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 1. 添加用户消息到记忆
        self.memory.add_message("user", user_input)
        
        # 2. 构建完整的上下文
        messages = self._build_context()
        
        # 3. 调用LLM
        response = self.llm.chat(messages)
        
        # 4. 添加助手回复到记忆
        self.memory.add_message("assistant", response)
        
        return response
    
    def _build_context(self) -> List[Dict[str, str]]:
        """
        构建上下文
        
        Returns:
            包含system prompt和历史消息的完整上下文
        """
        # System prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # 添加历史消息
        messages.extend(self.memory.get_messages())
        
        return messages
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.memory.get_messages()
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.memory.clear()
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        messages = self._build_context()
        total_tokens = self.llm.count_messages_tokens(messages)
        
        return {
            "turns": self.memory.get_turn_count(),
            "messages": len(self.memory),
            "total_tokens": total_tokens,
            "is_full": self.memory.is_full()
        }

class MemoryChatbotWithCompressor:
    """
    带记忆和压缩功能的聊天机器人
    
    特点:
    - 使用短期记忆保存对话历史
    - 自动管理上下文
    - 自动管理上下文
    """
    
    def __init__(self, 
                 llm: BaseLLM,
                 system_prompt: Optional[str] = None,
                 max_turns: int = 10,
                 compressor: Optional[BaseCompressor] = None,
                 enable_compression: bool = False,
                 compression_trigger: int = 10):
        """
        初始化
        
        Args:
            llm: LLM实例
            system_prompt: 系统提示词
            max_turns: 最大保存轮数
            compressor: 压缩器实例（如果为None，使用默认的混合策略）
            enable_compression: 是否启用压缩
            compression_trigger: 触发压缩的轮数阈值
        """
        self.llm = llm
        self.system_prompt = system_prompt or "你是一个友好、乐于助人的AI助手。"
        
        # 初始化短期记忆
        self.memory = ShortTermMemory(max_turns=max_turns)

        # 压缩配置
        self.enable_compression = enable_compression
        self.compression_trigger = compression_trigger

        # 初始化压缩器
        if compressor is None and enable_compression:
            # 默认使用混合策略
            self.compressor = HybridCompressor(
                llm, 
                threshold_turns=compression_trigger,
                keep_recent_turns=3
            )
        else:
            self.compressor = compressor
        
        # 统计信息
        self.stats = {
            "total_turns": 0,
            "compressions": 0,
            "tokens_saved": 0
        }
    
    def chat(self, user_input: str) -> str:
        """
        对话（带记忆）
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        # 1. 添加用户消息到记忆
        self.memory.add_message("user", user_input)
        
        # 2. 构建完整的上下文
        messages = self._build_context()
        
        # 3. 调用LLM
        response = self.llm.chat(messages)
        
        # 4. 添加助手回复到记忆
        self.memory.add_message("assistant", response)
        
        # 5. 更新统计
        self.stats["total_turns"] += 1

        return response
    
    def _build_context(self) -> List[Dict[str, str]]:
        """
        构建上下文(带压缩功能)
        
        Returns:
            包含system prompt和历史消息的完整上下文
        """
        # System prompt
        context = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # 获取历史消息
        history = self.memory.get_messages()
        
        # 判断是否需要压缩
        if self.enable_compression and self._should_compress(history):
            # 压缩前的token数
            tokens_before = self.llm.count_messages_tokens(history)

            # 执行压缩
            compressed_history = self.compressor.compress(history)

            # 压缩后的token数
            tokens_after = self.llm.count_messages_tokens(compressed_history)

            # 更新统计
            self.stats["compressions"] += 1
            self.stats["tokens_saved"] += (tokens_before - tokens_after)

            # 使用压缩后的历史
            context.extend(compressed_history)
        else:
            # 不压缩
            context.extend(history)

        return context

    def _should_compress(self, messages: List[Dict[str, str]]) -> bool:
        """
        判断是否应该触发压缩
        
        Args:
            messages: 消息列表
            
        Returns:
            是否应该压缩
        """
        # 根据轮数判断
        turn_count = len(messages) // 2
        return turn_count >= self.compression_trigger

    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.memory.get_messages()
    
    def clear_history(self) -> None:
        """清空对话历史"""
        self.memory.clear()
        self.stats = {
            "total_turns": 0,
            "compressions": 0,
            "tokens_saved": 0
        }
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        messages = self._build_context()
        total_tokens = self.llm.count_messages_tokens(messages)
        
        return {
            "turns": self.memory.get_turn_count(),
            "messages": len(self.memory),
            "total_tokens": total_tokens,
            "is_full": self.memory.is_full(),
            "compression_enabled": self.enable_compression,
            "total_compressions": self.stats["compressions"],
            "tokens_saved": self.stats["tokens_saved"],
            "compressor": self.compressor.__class__.__name__ if self.compressor else "None"
        }

    def set_compressor(self, compressor: BaseCompressor) -> None:
        """
        动态切换压缩器
        
        Args:
            compressor: 新的压缩器实例
        """
        self.compressor = compressor
        self.enable_compression = True


# ==================== 测试代码 ====================

def test_chatbot_with_compression():
    """测试带压缩功能的Chatbot"""
    from dotenv import load_dotenv
    from llm.deepseek import DeepSeekLLM
    
    load_dotenv()
    
    print("=" * 60)
    print("测试: 带压缩功能的Chatbot")
    print("=" * 60)
    
    # 初始化LLM
    llm = DeepSeekLLM()
    
    # 测试1: 不启用压缩
    print("\n场景1: 不启用压缩")
    print("-" * 60)
    bot1 = MemoryChatbotWithCompressor(llm, max_turns=10, enable_compression=False)
    
    for i in range(5):
        response = bot1.chat(f"这是第{i+1}条测试消息")
        print(f"Turn {i+1}: {response[:50]}...")
    
    stats1 = bot1.get_stats()
    print(f"\n统计信息:")
    print(f"  对话轮数: {stats1['turns']}")
    print(f"  总Token数: {stats1['total_tokens']}")
    print(f"  压缩次数: {stats1['total_compressions']}")
    
    # 测试2: 启用压缩（混合策略）
    print("\n\n场景2: 启用压缩（混合策略，阈值=3轮）")
    print("-" * 60)
    bot2 = MemoryChatbot(
        llm, 
        max_turns=10, 
        enable_compression=True,
        compression_trigger=3  # 3轮后触发压缩
    )
    
    for i in range(6):
        response = bot2.chat(f"这是第{i+1}条测试消息，我在测试压缩功能")
        print(f"Turn {i+1}: {response[:50]}...")
        
        if i == 2:
            print("  ⚠️  下一轮将触发压缩...")
        elif i == 3:
            print("  ✅ 已触发压缩!")
    
    stats2 = bot2.get_stats()
    print(f"\n统计信息:")
    print(f"  对话轮数: {stats2['turns']}")
    print(f"  总Token数: {stats2['total_tokens']}")
    print(f"  压缩次数: {stats2['total_compressions']}")
    print(f"  节省Token数: {stats2['tokens_saved']}")
    print(f"  压缩器: {stats2['compressor']}")
    
    # 测试3: 切换不同的压缩器
    print("\n\n场景3: 动态切换压缩器")
    print("-" * 60)
    bot3 = MemoryChatbot(llm, enable_compression=True)
    
    # 初始：使用混合策略
    print(f"初始压缩器: {bot3.compressor.__class__.__name__}")
    
    # 切换到滑动窗口
    bot3.set_compressor(SlidingWindowCompressor(keep_turns=3))
    print(f"切换后: {bot3.compressor.__class__.__name__}")
    
    for i in range(5):
        bot3.chat(f"测试消息 {i+1}")
    
    stats3 = bot3.get_stats()
    print(f"\n统计信息:")
    print(f"  压缩器: {stats3['compressor']}")
    print(f"  压缩次数: {stats3['total_compressions']}")


if __name__ == "__main__":
    test_chatbot_with_compression()