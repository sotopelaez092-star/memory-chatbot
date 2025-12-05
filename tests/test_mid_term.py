"""
中期记忆测试（适配现有PostgreSQLStorage）
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.database import DatabaseManager
from src.memory.mid_term import MidTermMemory


# 数据库配置
DB_USER = "postgres"
DB_PASSWORD = "password"  # 改成你的密码
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "test_memory"

DB_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def ensure_database():
    """确保测试数据库存在"""
    conn = await asyncpg.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database='postgres'
    )
    
    try:
        exists = await conn.fetchval(
            f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}'"
        )
        
        if not exists:
            await conn.execute(f'CREATE DATABASE {DB_NAME}')
            print(f"✓ 创建数据库 {DB_NAME}")
        else:
            print(f"✓ 数据库 {DB_NAME} 已存在")
    finally:
        await conn.close()


async def test_basic_flow():
    """
    测试1：基本流程
    - 添加消息
    - 触发溢出
    - 保存到PostgreSQL
    """
    print("\n" + "=" * 60)
    print("测试1：基本流程")
    print("=" * 60)
    
    # 初始化数据库
    db_manager = DatabaseManager(DB_URL)
    await db_manager.create_tables()
    print("✓ 数据库表已创建")
    
    # 创建session
    engine = create_async_engine(DB_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = async_session_maker()
    
    # 创建storage和memory
    storage = PostgreSQLStorage(session)
    memory = MidTermMemory(storage, max_turns=3)  # 3轮=6条消息
    
    # 添加3轮对话（刚好满）
    print("\n添加3轮对话...")
    for i in range(1, 4):
        await memory.add_message("user1", "session1", "user", f"消息{i}", tokens=10)
        await memory.add_message("user1", "session1", "assistant", f"回复{i}", tokens=10)
    
    print(f"  短期记忆: {len(memory.short_term)} 条")
    
    # 添加第4轮（触发溢出）
    print("\n添加第4轮（触发溢出）...")
    await memory.add_message("user1", "session1", "user", "消息4", tokens=10)
    await memory.add_message("user1", "session1", "assistant", "回复4", tokens=10)
    
    print(f"  短期记忆: {len(memory.short_term)} 条")
    
    # 检查PostgreSQL
    conv = await storage.get_or_create_conversation("user1", "session1")
    all_msgs = await storage.query_messages(conv.id)
    print(f"  PostgreSQL: {len(all_msgs)} 条")
    
    # 关闭
    await session.close()
    await engine.dispose()
    
    print("\n✅ 测试1完成")


async def test_compression():
    """
    测试2：压缩功能
    - 添加50条消息
    - 触发压缩
    - 检查摘要和画像
    """
    print("\n" + "=" * 60)
    print("测试2：压缩功能")
    print("=" * 60)
    
    # 创建新session
    engine = create_async_engine(DB_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = async_session_maker()
    
    storage = PostgreSQLStorage(session)
    memory = MidTermMemory(storage, max_turns=5)
    
    # 添加50条消息
    print("\n添加50条消息...")
    for i in range(1, 26):
        await memory.add_message("user2", "session2", "user", f"我叫Tom，第{i}条", tokens=10)
        await memory.add_message("user2", "session2", "assistant", f"知道了，第{i}条", tokens=10)
        
        if i % 10 == 0:
            print(f"  进度: {i*2}/50")
    
    # 检查压缩结果
    conv = await storage.get_or_create_conversation("user2", "session2")
    summaries = await storage.get_summaries(conv.id)
    profiles = await storage.get_profile(conv.id)
    
    print(f"\n压缩结果:")
    print(f"  摘要数量: {len(summaries)}")
    if summaries:
        for s in summaries:
            print(f"    - {s.time_range}")
            print(f"      {s.summary_text[:60]}...")
    
    print(f"\n  用户画像: {len(profiles) if profiles else 0} 个字段")
    if profiles:
        for k, v in profiles.items():
            print(f"    - {k}: {v}")
    
    # 关闭
    await session.close()
    await engine.dispose()
    
    print("\n✅ 测试2完成")


async def test_context_retrieval():
    """
    测试3：上下文获取
    - 不使用压缩
    - 使用压缩
    """
    print("\n" + "=" * 60)
    print("测试3：上下文获取")
    print("=" * 60)
    
    # 创建session
    engine = create_async_engine(DB_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = async_session_maker()
    
    storage = PostgreSQLStorage(session)
    memory = MidTermMemory(storage, max_turns=5)
    
    # 获取上下文（不使用压缩）
    print("\n获取上下文（不使用压缩）...")
    context1 = await memory.get_context_for_llm(
        "user2", "session2",
        use_compression=False
    )
    print(f"  返回消息数: {len(context1)}")
    
    # 获取上下文（使用压缩）
    print("\n获取上下文（使用压缩）...")
    context2 = await memory.get_context_for_llm(
        "user2", "session2",
        use_compression=True
    )
    print(f"  返回消息数: {len(context2)}")
    
    # 显示前3条
    if len(context2) >= 3:
        print("\n  前3条消息:")
        for i, msg in enumerate(context2[:3]):
            preview = msg['content'][:40] + "..." if len(msg['content']) > 40 else msg['content']
            print(f"    {i+1}. [{msg['role']}] {preview}")
    
    # 关闭
    await session.close()
    await engine.dispose()
    
    print("\n✅ 测试3完成")


async def test_session_recovery():
    """
    测试4：会话恢复
    - 清空短期记忆
    - 从PostgreSQL恢复
    """
    print("\n" + "=" * 60)
    print("测试4：会话恢复")
    print("=" * 60)
    
    # 创建session
    engine = create_async_engine(DB_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = async_session_maker()
    
    storage = PostgreSQLStorage(session)
    memory = MidTermMemory(storage, max_turns=5)
    
    # 清空
    print("\n清空短期记忆...")
    await memory.clear_session("user2", "session2")
    print(f"  清空后: {len(memory.short_term)} 条")
    
    # 恢复
    print("\n从PostgreSQL恢复...")
    await memory.load_recent_history("user2", "session2", count=10)
    print(f"  恢复后: {len(memory.short_term)} 条")
    
    # 关闭
    await session.close()
    await engine.dispose()
    
    print("\n✅ 测试4完成")


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试中期记忆")
    print("=" * 60)
    
    try:
        # 确保数据库存在
        await ensure_database()
        
        # 运行测试
        await test_basic_flow()
        await test_compression()
        await test_context_retrieval()
        await test_session_recovery()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n提示：测试数据保留在数据库中")
    print(f"清理命令：psql -U {DB_USER} -c 'DROP DATABASE {DB_NAME};'")


if __name__ == "__main__":
    asyncio.run(main())