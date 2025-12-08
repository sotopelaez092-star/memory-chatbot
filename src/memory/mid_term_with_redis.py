"""
中期记忆管理器（带Redis缓存）

缓存策略：
1. 读取：Redis → PostgreSQL
2. 写入：同时写PostgreSQL和Redis
3. TTL：根据访问频率动态调整
"""

from typing import List, Dict, Optional
from .short_term import ShortTermMemory
from .postgres_storage import PostgreSQLStorage
from .redis_storage import RedisStorage


class MidTermMemoryWithRedis:
    """中期记忆管理器（Redis缓存版）"""

    def __init__(
        self,
        pg_storage: PostgreSQLStorage,
        redis_storage: RedisStorage,
        max_turns: int = 10,
        cache_ttl: int = 3600
    ):
        """
        初始化
        
        Args:
            pg_storage: PostgreSQL存储
            redis_storage: Redis缓存
            max_turns: 短期记忆最大轮数
            cache_ttl: 缓存过期时间（秒）
        """
        self.pg_storage = pg_storage
        self.redis = redis_storage
        self.short_term = ShortTermMemory(max_turns=max_turns)
        self.cache_ttl = cache_ttl
        
        # 统计
        self.cache_hits = 0
        self.cache_misses = 0

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None
    ) -> None:
        """添加消息（带Redis缓存）"""
        
        # 1. 添加到短期记忆
        self.short_term.add_message(role, content)
        
        # 2. 检查是否溢出
        overflow = self.short_term.check_overflow()
        
        # 3. 如果溢出，保存到PostgreSQL和Redis
        if overflow:
            # 获取会话
            conv = await self.pg_storage.get_or_create_conversation(
                user_id=user_id,
                session_id=session_id
            )
            
            # 保存溢出的消息到PostgreSQL
            await self.pg_storage.add_messages(conv.id, overflow)
            
            # 【新增】更新Redis缓存
            all_messages = await self.pg_storage.query_messages(
                conversation_id=conv.id
            )
            
            messages_dict = [
                {"role": m.role, "content": m.content}
                for m in all_messages
            ]
            
            await self.redis.cache_messages(
                user_id=user_id,
                session_id=session_id,
                messages=messages_dict,
                ttl=self.cache_ttl
            )
            
            print(f"✓ 溢出 {len(overflow)} 条消息到PostgreSQL + Redis")

    async def query_messages(
        self,
        user_id: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        查询消息（带缓存）
        
        读取策略：
        1. 先查Redis
        2. 缓存未命中，查PostgreSQL
        3. 查到后写入Redis
        """
        
        # 1. 先查Redis
        cached = await self.redis.get_cached_messages(user_id, session_id)
        
        if cached is not None:
            # 缓存命中
            self.cache_hits += 1
            
            if limit:
                cached = cached[-limit:]
            return cached
        
        # 2. 缓存未命中，查PostgreSQL
        self.cache_misses += 1
        
        conv = await self.pg_storage.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id
        )
        
        pg_messages = await self.pg_storage.query_messages(
            conversation_id=conv.id,
            limit=limit
        )
        
        messages_dict = [
            {"role": m.role, "content": m.content}
            for m in pg_messages
        ]
        
        # 3. 写入Redis缓存
        if messages_dict:
            await self.redis.cache_messages(
                user_id=user_id,
                session_id=session_id,
                messages=messages_dict,
                ttl=self.cache_ttl
            )
        
        return messages_dict

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def get_short_term_count(self) -> int:
        """获取短期记忆中的消息数量"""
        return len(self.short_term)