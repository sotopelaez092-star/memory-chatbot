"""
性能测试 - 建立基准（修复并发版）
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
import time
import os
from typing import List, Dict
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.database import DatabaseManager
from src.memory.mid_term import MidTermMemory


# 数据库配置
username = os.getenv("USER")
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "memory_chatbot_test"

DB_URL = f"postgresql+asyncpg://{username}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


async def ensure_database():
    """确保测试数据库存在"""
    conn = await asyncpg.connect(
        user=username,
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


class PerformanceTest:
    """性能测试类"""
    
    def __init__(self):
        self.results = {}
        self.async_session_maker = None
        self.engine = None
    
    async def setup(self):
        """初始化测试环境"""
        # 创建数据库表
        db_manager = DatabaseManager(DB_URL)
        try:
            await db_manager.drop_tables()
        except:
            pass
        await db_manager.create_tables()
        print("✓ 数据库表已创建")
        
        # 创建engine和session工厂
        self.engine = create_async_engine(DB_URL, echo=False)
        self.async_session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def teardown(self):
        """清理"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_add_message_speed(self, count: int = 100):
        """
        测试1：添加消息的速度
        
        指标：每秒能添加多少条消息
        """
        print(f"\n【测试1】添加 {count} 条消息的速度")
        print("-" * 50)
        
        # 创建独立session
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        memory = MidTermMemory(storage, max_turns=10)
        
        try:
            user_id = "perf_user"
            session_id = "perf_session"
            
            start_time = time.time()
            
            for i in range(count):
                role = "user" if i % 2 == 0 else "assistant"
                await memory.add_message(
                    user_id, session_id, role, f"消息{i}", tokens=10
                )
            
            elapsed = time.time() - start_time
            speed = count / elapsed
            
            print(f"  总耗时: {elapsed:.2f} 秒")
            print(f"  速度: {speed:.2f} 条/秒")
            print(f"  平均延迟: {elapsed/count*1000:.2f} ms/条")
            
            self.results['add_message'] = {
                'total_time': elapsed,
                'speed': speed,
                'avg_latency': elapsed/count*1000
            }
            
            return elapsed, speed
        
        finally:
            await session.close()
    
    async def test_overflow_performance(self):
        """
        测试2：溢出性能
        
        指标：触发溢出时的延迟
        """
        print(f"\n【测试2】溢出性能")
        print("-" * 50)
        
        # 创建独立session
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        memory = MidTermMemory(storage, max_turns=3)
        
        try:
            user_id = "overflow_user"
            session_id = "overflow_session"
            
            # 先填满短期记忆
            for i in range(6):
                role = "user" if i % 2 == 0 else "assistant"
                await memory.add_message(user_id, session_id, role, f"消息{i}")
            
            # 测试触发溢出的延迟
            overflow_times = []
            for i in range(10):
                start = time.time()
                await memory.add_message(user_id, session_id, "user", f"溢出{i}")
                overflow_times.append(time.time() - start)
            
            avg_overflow_time = sum(overflow_times) / len(overflow_times)
            
            print(f"  平均溢出延迟: {avg_overflow_time*1000:.2f} ms")
            print(f"  最大延迟: {max(overflow_times)*1000:.2f} ms")
            print(f"  最小延迟: {min(overflow_times)*1000:.2f} ms")
            
            self.results['overflow'] = {
                'avg_latency': avg_overflow_time*1000,
                'max_latency': max(overflow_times)*1000,
                'min_latency': min(overflow_times)*1000
            }
        
        finally:
            await session.close()
    
    async def test_compression_performance(self):
        """
        测试3：压缩性能
        
        指标：触发压缩时的延迟
        """
        print(f"\n【测试3】压缩性能")
        print("-" * 50)
        
        # 创建独立session
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        memory = MidTermMemory(storage, max_turns=10)
        
        try:
            user_id = "compress_user"
            session_id = "compress_session"
            
            # 添加48条（接近50）
            for i in range(48):
                role = "user" if i % 2 == 0 else "assistant"
                await memory.add_message(user_id, session_id, role, f"消息{i}")
            
            # 测试触发压缩的延迟（第49、50条会触发）
            start = time.time()
            await memory.add_message(user_id, session_id, "user", "触发压缩1")
            await memory.add_message(user_id, session_id, "assistant", "触发压缩2")
            compression_time = time.time() - start
            
            print(f"  压缩延迟: {compression_time*1000:.2f} ms")
            
            self.results['compression'] = {
                'latency': compression_time*1000
            }
        
        finally:
            await session.close()
    
    async def test_context_retrieval_speed(self):
        """
        测试4：上下文获取速度
        
        指标：获取上下文的延迟
        """
        print(f"\n【测试4】上下文获取速度")
        print("-" * 50)
        
        # 创建独立session
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        memory = MidTermMemory(storage, max_turns=10)
        
        try:
            user_id = "perf_user"
            session_id = "perf_session"
            
            # 测试不使用压缩
            times_no_compression = []
            for _ in range(20):
                start = time.time()
                await memory.get_context_for_llm(
                    user_id, session_id, use_compression=False
                )
                times_no_compression.append(time.time() - start)
            
            avg_no_comp = sum(times_no_compression) / len(times_no_compression)
            
            # 测试使用压缩
            times_with_compression = []
            for _ in range(20):
                start = time.time()
                await memory.get_context_for_llm(
                    user_id, session_id, use_compression=True
                )
                times_with_compression.append(time.time() - start)
            
            avg_with_comp = sum(times_with_compression) / len(times_with_compression)
            
            print(f"  不使用压缩: {avg_no_comp*1000:.2f} ms")
            print(f"  使用压缩: {avg_with_comp*1000:.2f} ms")
            
            self.results['context_retrieval'] = {
                'no_compression': avg_no_comp*1000,
                'with_compression': avg_with_comp*1000
            }
        
        finally:
            await session.close()
    
    async def test_concurrent_users(self, num_users: int = 10):
        """
        测试5：并发用户（修复版）
        
        模拟真实Web应用中多个用户同时发送请求的场景
        每个用户 = 一个HTTP请求 = 一个独立session
        """
        print(f"\n【测试5】并发 {num_users} 个用户")
        print("-" * 50)
        
        async def simulate_user(user_id: str):
            """
            模拟一个用户的请求
            
            真实场景对应：
            1. 用户发送HTTP请求
            2. FastAPI创建session（依赖注入）
            3. 处理业务逻辑
            4. 关闭session
            """
            # 每个请求创建独立session
            session = self.async_session_maker()
            storage = PostgreSQLStorage(session)
            memory = MidTermMemory(storage, max_turns=10)
            
            try:
                # 模拟用户发送10条消息
                for i in range(10):
                    role = "user" if i % 2 == 0 else "assistant"
                    await memory.add_message(
                        user_id, f"session_{user_id}", role, f"消息{i}"
                    )
            finally:
                await session.close()
        
        start = time.time()
        
        # 并发执行（模拟多个请求同时到达）
        tasks = [simulate_user(f"user_{i}") for i in range(num_users)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        total_messages = num_users * 10
        
        print(f"  总耗时: {elapsed:.2f} 秒")
        print(f"  总消息数: {total_messages}")
        print(f"  吞吐量: {total_messages/elapsed:.2f} 条/秒")
        print(f"  平均每用户: {elapsed/num_users:.2f} 秒")
        
        self.results['concurrent'] = {
            'total_time': elapsed,
            'throughput': total_messages/elapsed,
            'avg_per_user': elapsed/num_users
        }
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 60)
        print("性能测试摘要")
        print("=" * 60)
        
        print("\n【关键指标】")
        print(f"  添加消息速度: {self.results['add_message']['speed']:.2f} 条/秒")
        print(f"  添加消息延迟: {self.results['add_message']['avg_latency']:.2f} ms")
        print(f"  溢出延迟: {self.results['overflow']['avg_latency']:.2f} ms")
        print(f"  压缩延迟: {self.results['compression']['latency']:.2f} ms")
        print(f"  上下文获取（无压缩）: {self.results['context_retrieval']['no_compression']:.2f} ms")
        print(f"  上下文获取（有压缩）: {self.results['context_retrieval']['with_compression']:.2f} ms")
        print(f"  并发吞吐量: {self.results['concurrent']['throughput']:.2f} 条/秒")
        print(f"  平均每用户响应时间: {self.results['concurrent']['avg_per_user']*1000:.2f} ms")
        
        print("\n【性能评估】")
        if self.results['add_message']['speed'] > 50:
            print("  ✅ 添加消息速度：优秀")
        elif self.results['add_message']['speed'] > 20:
            print("  ⚠️  添加消息速度：良好，可优化")
        else:
            print("  ❌ 添加消息速度：需要优化")
        
        if self.results['context_retrieval']['no_compression'] < 50:
            print("  ✅ 上下文获取：优秀")
        elif self.results['context_retrieval']['no_compression'] < 100:
            print("  ⚠️  上下文获取：良好，可优化")
        else:
            print("  ❌ 上下文获取：需要优化")
        
        if self.results['concurrent']['throughput'] > 50:
            print("  ✅ 并发性能：优秀")
        elif self.results['concurrent']['throughput'] > 20:
            print("  ⚠️  并发性能：良好，可优化")
        else:
            print("  ❌ 并发性能：需要优化")
        
        print("\n【瓶颈分析】")
        if self.results['overflow']['avg_latency'] > 50:
            print("  🔍 溢出操作较慢，建议优化数据库写入")
        
        if self.results['compression']['latency'] > 100:
            print("  🔍 压缩操作较慢，建议异步化或优化LLM调用")
        
        if self.results['context_retrieval']['no_compression'] > 100:
            print("  🔍 上下文获取较慢，建议加入Redis缓存")
        
        if self.results['concurrent']['avg_per_user'] > 1:
            print("  🔍 并发响应时间较长，建议优化连接池配置")
        
        print("\n【优化建议】")
        suggestions = []
        
        if self.results['add_message']['speed'] < 100:
            suggestions.append("1. 批量写入优化")
        
        if self.results['context_retrieval']['no_compression'] > 50:
            suggestions.append("2. 加入Redis缓存（会话ID、用户画像、摘要）")
        
        if self.results['compression']['latency'] > 50:
            suggestions.append("3. 异步压缩（不阻塞用户）")
        
        if self.results['concurrent']['throughput'] < 100:
            suggestions.append("4. 数据库连接池调优")
        
        if suggestions:
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            print("  🎉 当前性能已经非常优秀，无需优化！")
        
        print("\n" + "=" * 60)


async def main():
    """运行性能测试"""
    print("=" * 60)
    print("中期记忆性能测试")
    print("=" * 60)
    print(f"数据库: {DB_URL}")
    
    # 确保数据库存在
    await ensure_database()
    
    test = PerformanceTest()
    
    try:
        # 初始化
        print("\n初始化测试环境...")
        await test.setup()
        
        # 运行测试
        await test.test_add_message_speed(count=100)
        await test.test_overflow_performance()
        await test.test_compression_performance()
        await test.test_context_retrieval_speed()
        await test.test_concurrent_users(num_users=10)
        
        # 打印摘要
        test.print_summary()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await test.teardown()
        print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())