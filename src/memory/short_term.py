from typing import List, Dict, Optional
from collections import deque

class ShortTermMemory:
    """
    短期记忆（内存）
    
    特点:
    - 存储在内存中（速度极快）
    - 容量有限（最近N轮对话）
    - 会话结束即清空
    """

    def __init__(self, max_turns: int = 10):
        """
        初始化短期记忆
        
        Args:
            max_turns: 最大保存轮数（1轮 = user + assistant）
        """
        self.max_turns = max_turns
        self.max_messages = max_turns * 2  # 每轮包含user和assistant两条消息
        
        # 使用deque实现固定大小的队列
        self.messages: deque = deque(maxlen=self.max_messages)

    def add_message(self, role: str, content: str) -> None:
        """
        添加一条消息
        
        Args:
            role: 'user' 或 'assistant' 或 'system'
            content: 消息内容
        """
        message = {
            "role": role,
            "content": content
        }
        self.messages.append(message)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        获取所有消息
        
        Returns:
            消息列表
        """
        return list(self.messages)

    def get_recent_messages(self, n: int) -> List[Dict[str, str]]:
        """
        获取最近的N条消息
        
        Args:
            n: 消息数量
            
        Returns:
            最近的N条消息
        """
        return list(self.messages)[-n:] if n > 0 else []

    def clear(self) -> None:
        """清空所有消息"""
        self.messages.clear()

    def get_turn_count(self) -> int:
        """
        获取当前轮数
        
        Returns:
            轮数（消息数 / 2）
        """
        return len(self.messages) // 2

    def is_full(self) -> bool:
        """
        判断是否已满
        
        Returns:
            是否已满
        """
        return len(self.messages) >= self.max_messages

    def __len__(self) -> int:
        """返回当前消息数量"""
        return len(self.messages)

    def __repr__(self) -> str:
        return f"ShortTermMemory(messages={len(self.messages)}, max={self.max_messages})"


# 测试代码
if __name__ == "__main__":
    memory = ShortTermMemory(max_turns=3)
    
    print("=== 测试短期记忆 ===\n")
    
    # 添加对话
    memory.add_message("user", "我叫Tom")
    memory.add_message("assistant", "你好Tom！")
    
    memory.add_message("user", "我喜欢吃辣的")
    memory.add_message("assistant", "记住了，你喜欢吃辣的")
    
    memory.add_message("user", "我在上海工作")
    memory.add_message("assistant", "上海是个好地方")
    
    print(f"当前轮数: {memory.get_turn_count()}")
    print(f"是否已满: {memory.is_full()}")
    print(f"\n当前消息:")
    for msg in memory.get_messages():
        print(f"  {msg['role']}: {msg['content']}")
    
    # 再添加一轮 - 会自动删除最早的
    print("\n=== 添加第4轮（超出容量）===\n")
    memory.add_message("user", "推荐个餐厅")
    memory.add_message("assistant", "推荐川菜馆")
    
    print(f"当前轮数: {memory.get_turn_count()}")
    print(f"\n当前消息:")
    for msg in memory.get_messages():
        print(f"  {msg['role']}: {msg['content']}")
    
    print("\n注意：第1轮对话（'我叫Tom'）已被删除！")
