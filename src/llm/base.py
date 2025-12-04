from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseLLM(ABC):
    """LLM抽象基类"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: float = 0.7) -> str:
        f"""
        对话接口

        Args:
            messages: 消息列表[{"role":"user", "content": "..."}, ...]
            max_tokens: 最大token数
            temperature: 温度参数
        
        Returns:
            assistant的回复
        """
        pass

    def count_tokens(self, text: str) -> int:
        """计算文本的token数"""
        pass

