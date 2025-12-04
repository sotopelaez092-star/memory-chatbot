import os
from typing import List, Dict, Optional
from openai import OpenAI
import tiktoken

from .base import BaseLLM

class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM实现"""

    def __init__(self,
        api_key: Optional[str] = None,
        base_url:Optional[str] = None,
        model: str = "deepseek-chat"):
        """
        初始化DeepSeek客户端

        Args:
            api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
            model: 使用的模型名
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        
        # 创建OpenAI客户端（DeepSeek兼容OpenAI接口）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Token计数器
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    
    def chat(self, messages: List[Dict[str, str]],
            max_tokens: Optional[int] = None,
            temperature: float = 0.7) -> str:
        """
        发送对话请求
        
        Args:
            messages: 消息列表
            max_tokens: 最大token数
            temperature: 温度参数
            
        Returns:
            助手的回复
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"LLM调用失败: {str(e)}")
    

    def count_tokens(self, text: str) -> int:
        """计算token数"""
        return len(self.encoding.encode(text))

    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的总token数"""
        total = 0
        for message in messages:
            # 每条消息有固定的格式开销
            total += 4  # role和content的分隔符
            total += self.count_tokens(message.get("role", ""))
            total += self.count_tokens(message.get("content", ""))
        total += 2  # 每个请求的开始和结束
        return total


# 测试代码
if __name__ == "__main__":
    llm = DeepSeekLLM()
    
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
    
    response = llm.chat(messages)
    print(f"回复: {response}")
    print(f"Token数: {llm.count_tokens(response)}")