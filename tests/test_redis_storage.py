"""
测试RedisStorage类
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.redis_storage import RedisStorage


async def test_message_cache():
    """测试消息缓存"""
    print("=" * 60)
    print("测试消息缓存功能")
    print("=" * 60)
    
    storage = RedisStorage()
    
    try:
        await storage.connect()
        
        user_id = "test_user"
        session_id = "test_session"
        
        # 测试1：缓存消息
        print("\n【测试1】缓存消息")
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
            {"role": "user", "content": "今天天气怎么样"},
            {"role": "assistant", "content": "今天天气不错！"}
        ]
        
        await storage.cache_messages(user_id, session_id, messages, ttl=60)
        
        # 测试2：读取缓存（应该命中）
        print("\n【测试2】读取缓存（应该命中）")
        cached = await storage.get_cached_messages(user_id, session_id)
        
        if cached:
            print(f"  ✓ 读取到 {len(cached)} 条消息")
            for msg in cached:
                print(f"    - {msg['role']}: {msg['content']}")
        
        # 测试3：读取不存在的缓存（应该未命中）
        print("\n【测试3】读取不存在的缓存（应该未命中）")
        cached2 = await storage.get_cached_messages("other_user", "other_session")
        
        if cached2 is None:
            print("  ✓ 正确返回None")
        
        # 测试4：查看TTL
        print("\n【测试4】查看TTL")
        key = storage._message_list_key(user_id, session_id)
        ttl = await storage.redis.ttl(key)
        print(f"  ✓ 剩余TTL: {ttl}秒")
        
        print("\n✅ 消息缓存测试通过！")
        
    finally:
        # 清理测试数据
        await storage.redis.delete(storage._message_list_key(user_id, session_id))
        await storage.close()

async def test_profile_cache():
    """测试画像缓存"""
    print("\n" + "=" * 60)
    print("测试画像缓存功能")
    print("=" * 60)
    
    storage = RedisStorage()
    
    try:
        await storage.connect()
        
        user_id = "test_user"
        session_id = "test_session"
        
        # 测试1：缓存画像
        print("\n【测试1】缓存画像")
        profile = {
            "name": "Tom",
            "age": "28",
            "city": "Shanghai",
            "interests": "AI,编程,异步编程"
        }
        
        await storage.cache_profile(user_id, session_id, profile, ttl=120)
        
        # 测试2：读取缓存（应该命中）
        print("\n【测试2】读取画像（应该命中）")
        cached_profile = await storage.get_cached_profile(user_id, session_id)
        
        if cached_profile:
            print(f"  ✓ 读取到画像:")
            for key, value in cached_profile.items():
                print(f"    - {key}: {value}")
        
        # 测试3：读取不存在的画像（应该未命中）
        print("\n【测试3】读取不存在的画像（应该未命中）")
        cached2 = await storage.get_cached_profile("other_user", "other_session")
        
        if cached2 is None:
            print("  ✓ 正确返回None")
        
        print("\n✅ 画像缓存测试通过！")
        
    finally:
        # 清理测试数据
        await storage.redis.delete(storage._profile_key(user_id, session_id))
        await storage.close()


# 在 main 里调用
async def main():
    await test_message_cache()
    await test_profile_cache()


if __name__ == "__main__":
    asyncio.run(main())
