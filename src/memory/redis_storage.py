"""
Redis存储层

基于Sorted Set + Hash实现中期记忆存储
"""

import json
import time
from typing import List, Dict, Optional
import redis

from .lua_scripts import (
    ADD_MESSAGE_SCRIPT,
    GET_MESSAGES_SCRIPT,
    UPDATE_PROFILE_SCRIPT
)


class RedisStorage:
    """
    Redis存储管理器
    
    数据结构：
    - chat:{user_id}:{session_id}:messages  # Sorted Set，存储消息
    - chat:{user_id}:{session_id}:meta      # Hash，存储元数据
    - chat:{user_id}:{session_id}:profile   # Hash，存储用户画像
    - chat:{user_id}:{session_id}:summary   # Hash，存储历史摘要
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        max_messages: int = 50,
        ttl: int = 604800  # 7天
    ):
        """
        初始化
        
        Args:
            redis_client: Redis客户端
            max_messages: 最多保留消息数
            ttl: 过期时间（秒）
        """
        self.redis = redis_client
        self.max_messages = max_messages
        self.ttl = ttl
        
        # 注册Lua脚本（提高性能）
        self.add_message_sha = self.redis.script_load(ADD_MESSAGE_SCRIPT)
        self.get_messages_sha = self.redis.script_load(GET_MESSAGES_SCRIPT)
        self.update_profile_sha = self.redis.script_load(UPDATE_PROFILE_SCRIPT)
    
    def _get_key(self, user_id: str, session_id: str, suffix: str) -> str:
        """生成Redis key"""
        return f"chat:{user_id}:{session_id}:{suffix}"
    
    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[float] = None
    ) -> int:
        """
        添加消息（原子操作）
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            role: 角色（user/assistant/system）
            content: 消息内容
            timestamp: 时间戳（可选，默认当前时间）
        
        Returns:
            当前消息总数
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 构造消息对象
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        messages_key = self._get_key(user_id, session_id, "messages")
        
        # 使用Lua脚本原子性添加
        count = self.redis.evalsha(
            self.add_message_sha,
            1,  # key数量
            messages_key,  # KEYS[1]
            json.dumps(message, ensure_ascii=False),  # ARGV[1]
            timestamp,  # ARGV[2]
            self.max_messages,  # ARGV[3]
            self.ttl  # ARGV[4]
        )
        
        return count
    
    def get_messages(
        self,
        user_id: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        获取消息列表
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            limit: 获取数量（None=全部）
        
        Returns:
            消息列表
        """
        messages_key = self._get_key(user_id, session_id, "messages")
        meta_key = self._get_key(user_id, session_id, "meta")
        
        if limit is None:
            limit = self.max_messages
        
        # 使用Lua脚本获取消息 + 更新访问时间
        messages_json = self.redis.evalsha(
            self.get_messages_sha,
            2,  # key数量
            messages_key,  # KEYS[1]
            meta_key,  # KEYS[2]
            limit,  # ARGV[1]
            time.time()  # ARGV[2]
        )
        
        # 反序列化
        messages = [json.loads(msg) for msg in messages_json]
        return messages
    
    def update_profile(
        self,
        user_id: str,
        session_id: str,
        profile_data: Dict[str, str]
    ) -> bool:
        """
        更新用户画像
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            profile_data: 画像数据，例如 {"name": "Tom", "age": "28"}
        
        Returns:
            是否成功
        """
        profile_key = self._get_key(user_id, session_id, "profile")
        
        # 构造参数：[ttl, field1, value1, field2, value2, ...]
        args = [self.ttl]
        for field, value in profile_data.items():
            args.extend([field, str(value)])
        
        # 使用Lua脚本更新
        result = self.redis.evalsha(
            self.update_profile_sha,
            1,  # key数量
            profile_key,  # KEYS[1]
            *args  # ARGV[1], ARGV[2], ...
        )
        
        return result == 1
    
    def get_profile(
        self,
        user_id: str,
        session_id: str
    ) -> Dict[str, str]:
        """
        获取用户画像
        
        Returns:
            用户画像字典
        """
        profile_key = self._get_key(user_id, session_id, "profile")
        return self.redis.hgetall(profile_key)