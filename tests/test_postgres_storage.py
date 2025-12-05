"""
PostgreSQL存储层测试 - 最简单版本
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


username = os.getenv("USER")
TEST_DATABASE_URL = f"postgresql+asyncpg://{username}@localhost:5432/memory_chatbot_test"



async def test_add_message_async():
    """测试添加单条消息"""
    from src.memory.database import DatabaseManager
    from src.memory.postgres_storage import PostgreSQLStorage
    
    print("\n开始测试 test_add_message...")
    
    # 创建数据库
    manager = DatabaseManager(TEST_DATABASE_URL, echo=False)
    await manager.create_tables()
    print("✓ 表创建成功")
    
    try:
        # 创建storage
        async with manager.async_session_maker() as session:
            storage = PostgreSQLStorage(session)
            
            # 创建会话
            conv = await storage.get_or_create_conversation(
                user_id="msg_user_1",
                session_id="msg_session_1"
            )
            print(f"✓ 会话创建成功，ID: {conv.id}")
            
            # 添加消息
            msg = await storage.add_message(
                conversation_id=conv.id,
                role="user",
                content="你好",
                tokens=5
            )
            print(f"✓ 消息创建成功，ID: {msg.id}")
            
            # 验证
            assert msg.id is not None
            assert msg.role == "user"
            assert msg.content == "你好"
            assert msg.tokens == 5
            
            await session.commit()
            print("✅ test_add_message 通过！")
    
    finally:
        # 清理
        await manager.drop_tables()
        await manager.close()
        print("✓ 清理完成")


async def test_get_or_create_conversation_async():
    """测试获取或创建会话"""
    from src.memory.database import DatabaseManager
    from src.memory.postgres_storage import PostgreSQLStorage
    
    print("\n开始测试 test_get_or_create_conversation...")
    
    manager = DatabaseManager(TEST_DATABASE_URL, echo=False)
    await manager.create_tables()
    
    try:
        async with manager.async_session_maker() as session:
            storage = PostgreSQLStorage(session)
            
            # 第一次调用
            conv1 = await storage.get_or_create_conversation(
                user_id="test_user_123",
                session_id="test_session_456",
                title="测试会话"
            )
            print(f"✓ 第一次创建，ID: {conv1.id}")
            
            # 第二次调用
            conv2 = await storage.get_or_create_conversation(
                user_id="test_user_123",
                session_id="test_session_456"
            )
            print(f"✓ 第二次获取，ID: {conv2.id}")
            
            # 验证
            assert conv2.id == conv1.id
            
            await session.commit()
            print("✅ test_get_or_create_conversation 通过！")
    
    finally:
        await manager.drop_tables()
        await manager.close()


async def test_add_messages_async():
    """测试批量添加消息"""
    from src.memory.database import DatabaseManager
    from src.memory.postgres_storage import PostgreSQLStorage
    
    print("\n开始测试 test_add_messages...")
    
    manager = DatabaseManager(TEST_DATABASE_URL, echo=False)
    await manager.create_tables()
    
    try:
        async with manager.async_session_maker() as session:
            storage = PostgreSQLStorage(session)
            
            conv = await storage.get_or_create_conversation(
                user_id="batch_user",
                session_id="batch_session"
            )
            
            messages_list = [
                {"role": "user", "content": "你好", "tokens": 5},
                {"role": "assistant", "content": "你好！", "tokens": 4}
            ]
            
            messages = await storage.add_messages(conv.id, messages_list)
            
            assert len(messages) == 2
            assert messages[0].role == "user"
            print(f"✓ 批量添加了 {len(messages)} 条消息")
            
            await session.commit()
            print("✅ test_add_messages 通过！")
    
    finally:
        await manager.drop_tables()
        await manager.close()


async def test_get_tokens_async():
    """测试获取Token统计"""
    from src.memory.database import DatabaseManager
    from src.memory.postgres_storage import PostgreSQLStorage
    
    print("\n开始测试 test_get_tokens...")
    
    manager = DatabaseManager(TEST_DATABASE_URL, echo=False)
    await manager.create_tables()
    
    try:
        async with manager.async_session_maker() as session:
            storage = PostgreSQLStorage(session)
            
            conv = await storage.get_or_create_conversation(
                user_id="token_user",
                session_id="token_session"
            )
            
            messages_list = [
                {"role": "user", "content": "消息1", "tokens": 10},
                {"role": "assistant", "content": "消息2", "tokens": 20},
                {"role": "user", "content": "消息3", "tokens": 5}
            ]
            await storage.add_messages(conv.id, messages_list)
            
            tokens = await storage.get_tokens(conv.id)
            
            assert tokens["total"] == 35
            assert tokens["user"] == 15
            assert tokens["assistant"] == 20
            print(f"✓ Token统计: {tokens}")
            
            await session.commit()
            print("✅ test_get_tokens 通过！")
    
    finally:
        await manager.drop_tables()
        await manager.close()


async def test_upsert_profile_async():
    """测试upsert用户画像"""
    from src.memory.database import DatabaseManager
    from src.memory.postgres_storage import PostgreSQLStorage
    
    print("\n开始测试 test_upsert_profile...")
    
    manager = DatabaseManager(TEST_DATABASE_URL, echo=False)
    await manager.create_tables()
    
    try:
        async with manager.async_session_maker() as session:
            storage = PostgreSQLStorage(session)
            
            conv = await storage.get_or_create_conversation(
                user_id="profile_user",
                session_id="profile_session"
            )
            
            # 第一次插入
            await storage.upsert_profile(conv.id, {"name": "Tom", "age": "28"})
            profile = await storage.get_profile(conv.id)
            assert profile["name"] == "Tom"
            print(f"✓ 首次插入: {profile}")
            
            # 更新
            await storage.upsert_profile(conv.id, {"age": "29", "city": "上海"})
            profile = await storage.get_profile(conv.id)
            assert profile["age"] == "29"
            assert profile["name"] == "Tom"
            print(f"✓ 更新后: {profile}")
            
            await session.commit()
            print("✅ test_upsert_profile 通过！")
    
    finally:
        await manager.drop_tables()
        await manager.close()


async def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行所有测试")
    print("=" * 60)
    
    tests = [
        test_add_message_async,
        test_get_or_create_conversation_async,
        test_add_messages_async,
        test_get_tokens_async,
        test_upsert_profile_async
    ]
    
    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())