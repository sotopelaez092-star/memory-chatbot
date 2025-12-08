"""
Redis缓存存储
"""

import json
from typing import List, Dict, Optional
import redis.asyncio as aioredis

class RedisStorage:
    """Redis缓存存储"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600
    ):
        self.host = host
        self.port = port
        self.db = db
        self.default_ttl = default_ttl
        self.redis = None

    async def connect(self):
        """建立连接"""
        self.redis = await aioredis.from_url(
            f"redis://{self.host}:{self.port}/{self.db}",
            encoding="utf-8",
            decode_responses=True
        )

    async def close(self):
        """关闭连接"""
        if self.redis:
            await self.redis.close()
            self.redis = None

    def _message_list_key(self, user_id: str, session_id: str) -> str:
        """生成消息列表的缓存key"""
        # 返回格式：messages:{user_id}:{session_id}
        return f"messages:{user_id}:{session_id}"

    async def cache_messages(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict],
        ttl: Optional[int] = None
    ):
        """
        缓存消息列表
    
        Args:
            user_id: 用户ID
            session_id: 会话ID
            messages: 消息列表 [{"role": "user", "content": "..."}]
            ttl: 过期时间（秒），None使用默认值
        """
        # 1. 生成key
        key = self._message_list_key(user_id, session_id)
        
        # 2. 序列化为JSON
        json_messages = json.dumps(messages, ensure_ascii=False)
        
        # 3. 设置缓存（带TTL）
        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            json_messages
        )
        
        
        print(f"  ✓ 缓存消息: {len(messages)}条 (TTL={ttl or self.default_ttl}s)")

    async def get_cached_messages(
        self,
        user_id: str,
        session_id: str
    ) -> Optional[List[Dict]]:
        """
        获取缓存的消息
        
        Returns:
            消息列表，缓存未命中返回None
        """
        # 1. 生成key
        key = self._message_list_key(user_id, session_id)
        
        # 2. 从Redis读取
        value = await self.redis.get(key)
        
        # 3. 判断是否命中
        if value:
            # 反序列化JSON
            messages = json.loads(value)
            print(f"  ✓ 缓存命中: {len(messages)}条消息")
            return messages
        else:
            print(f"  ✗ 缓存未命中")
            return None

    def _profile_key(self, user_id: str, session_id: str) -> str:
        """生成用户画像的缓存key"""
        return f"profile:{user_id}:{session_id}"

    async def cache_profile(
        self,
        user_id: str,
        session_id: str,
        profile: Dict[str, str],
        ttl: Optional[int] = None
    ):
        """
        缓存用户画像
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            profile: 用户画像 {"name": "Tom", "age": "28"}
            ttl: 过期时间（秒）
        """
        # 1. 生成key
        key = self._profile_key(user_id, session_id)
        
        # 2. 设置Hash
        await self.redis.hset(key, mapping=profile)
        
        # 3. 设置过期时间
        await self.redis.expire(key, ttl or self.default_ttl)
        
        print(f"  ✓ 缓存画像: {list(profile.keys())}")

    async def get_cached_profile(
        self,
        user_id: str,
        session_id: str
    ) -> Optional[Dict[str, str]]:
        """
        获取缓存的用户画像
        
        Returns:
            用户画像，缓存未命中返回None
        """
        # 1. 生成key
        key = self._profile_key(user_id, session_id)
        
        # 2. 读取Hash
        profile = await self.redis.hgetall(key)
        
        # 3. 判断是否命中
        if profile:
            print(f"  ✓ 缓存命中: 画像 {list(profile.keys())}")
            return profile
        else:
            print(f"  ✗ 缓存未命中: 画像")
            return None