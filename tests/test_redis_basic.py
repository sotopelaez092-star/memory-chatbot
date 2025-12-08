"""
测试Redis基本连接
"""
import asyncio
import redis.asyncio as aioredis


async def test_redis_connection():
    """测试Redis连接"""
    print("=" * 60)
    print("测试Redis基本连接")
    print("=" * 60)
    
    # 连接Redis
    redis_client = await aioredis.from_url(
        "redis://localhost:6379/0",
        encoding="utf-8",
        decode_responses=True
    )
    
    try:
        # 测试1：Ping
        print("\n【测试1】Ping")
        result = await redis_client.ping()
        print(f"  ✓ Ping结果: {result}")
        
        # 测试2：Set/Get
        print("\n【测试2】Set/Get")
        await redis_client.set("test_key", "hello redis")
        value = await redis_client.get("test_key")
        print(f"  ✓ 设置值: test_key = 'hello redis'")
        print(f"  ✓ 读取值: {value}")
        
        # 测试3：带过期时间
        print("\n【测试3】Set with TTL")
        await redis_client.setex("temp_key", 10, "will expire in 10s")
        ttl = await redis_client.ttl("temp_key")
        print(f"  ✓ 设置临时key，TTL={ttl}秒")
        
        # 测试4：Hash
        print("\n【测试4】Hash操作")
        await redis_client.hset("user:1", mapping={
            "name": "Tom",
            "age": "28",
            "city": "Shanghai"
        })
        user = await redis_client.hgetall("user:1")
        print(f"  ✓ 设置用户信息: {user}")
        
        # 测试5：删除
        print("\n【测试5】删除key")
        await redis_client.delete("test_key", "temp_key", "user:1")
        print(f"  ✓ 已删除所有测试key")
        
        print("\n✅ Redis连接测试通过！")
        
    finally:
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(test_redis_connection())