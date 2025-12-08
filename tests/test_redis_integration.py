"""
测试Redis集成到中期记忆系统
"""
import asyncio
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.mid_term_with_redis import MidTermMemoryWithRedis
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.redis_storage import RedisStorage
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

username = os.getenv("USER")
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "memory_chatbot_test"

DB_URL = f"postgresql+asyncpg://{username}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

async def test_basic_functionality():
    """测试基础功能"""
    print("=" * 60)
    print("测试Redis集成基础功能")
    print("=" * 60)
    
    # 1. 创建PostgreSQL连接
    # 修改这里
    engine = create_async_engine(
        "postgresql+asyncpg://FiaShi@localhost/memory_chatbot_test",  # ← 改成这个
        echo=False
    )
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    session = async_session_maker()
    pg_storage = PostgreSQLStorage(session)
    
    # 2. 创建Redis连接
    redis_storage = RedisStorage()
    await redis_storage.connect()
    
    # 3. 创建中期记忆（带Redis）
    memory = MidTermMemoryWithRedis(
        pg_storage=pg_storage,
        redis_storage=redis_storage,
        max_turns=3,  # 短期记忆只保留5轮
        cache_ttl=300  # 缓存5分钟
    )
    
    try:
        user_id = "test_user"
        session_id = "test_session"
        
        # 测试1：添加消息
        print("\n【测试1】添加10条消息")
        for i in range(10):
            await memory.add_message(
                user_id, session_id,
                "user" if i % 2 == 0 else "assistant",
                f"消息内容 {i+1}"
            )
        
        # 测试2：第一次查询（应该未命中，从PostgreSQL读）
        print("\n【测试2】第一次查询（缓存未命中）")
        messages = await memory.query_messages(user_id, session_id)
        print(f"  查询到 {len(messages)} 条消息")
        print(f"  缓存命中率: {memory.get_cache_hit_rate():.2%}")
        
        # 测试3：第二次查询（应该命中，从Redis读）
        print("\n【测试3】第二次查询（缓存命中）")
        messages = await memory.query_messages(user_id, session_id)
        print(f"  查询到 {len(messages)} 条消息")
        print(f"  缓存命中率: {memory.get_cache_hit_rate():.2%}")
        
        # 测试4：第三次查询（应该命中）
        print("\n【测试4】第三次查询（缓存命中）")
        messages = await memory.query_messages(user_id, session_id)
        print(f"  查询到 {len(messages)} 条消息")
        print(f"  缓存命中率: {memory.get_cache_hit_rate():.2%}")
        
        print("\n✅ 基础功能测试通过！")
        
    finally:
        # 清理
        await redis_storage.redis.delete(
            redis_storage._message_list_key(user_id, session_id)
        )
        await session.close()
        await redis_storage.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())