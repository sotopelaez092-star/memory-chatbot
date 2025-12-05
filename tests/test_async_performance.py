"""
异步压缩性能对比测试（带详细API调用日志）

对比：
1. 同步压缩（阻塞用户） + 真实LLM
2. 异步压缩（不阻塞） + 真实LLM
"""

import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
import time
import httpx
import json
from typing import List, Dict
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.database import DatabaseManager
from src.memory.mid_term import MidTermMemory
from src.memory.mid_term_async import MidTermMemoryAsync

load_dotenv()

username = os.getenv("USER")
DB_NAME = "memory_chatbot_test"
DB_URL = f"postgresql+asyncpg://{username}@localhost:5432/{DB_NAME}"


# ✅ 继承方式：扩展同步版本支持真实LLM + 详细日志
class MidTermMemoryWithLLM(MidTermMemory):
    """扩展MidTermMemory，添加真实LLM压缩功能 + 详细日志"""
    
    def __init__(self, storage, max_turns: int = 10, enable_real_compression: bool = False):
        from src.memory.short_term import ShortTermMemory
        
        # 直接初始化，不调用父类__init__
        self.storage = storage
        self.short_term = ShortTermMemory(max_turns=max_turns)
        
        # 添加LLM配置
        self.enable_real_compression = enable_real_compression
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_base = "https://api.deepseek.com/v1"
        
        if self.enable_real_compression and not self.api_key:
            print("⚠️  警告：未设置DEEPSEEK_API_KEY，将使用假压缩")
            self.enable_real_compression = False
    
    async def _generate_summary(self, messages: List) -> str:
        """重写：支持真实LLM压缩 + 详细日志"""
        if not self.enable_real_compression:
            # 假压缩
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"
        
        # 真实LLM压缩
        try:
            conversation_text = ""
            for msg in messages:
                role = msg.role if hasattr(msg, 'role') else "user"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                conversation_text += f"{role}: {content}\n"
            
            prompt = f"""请用1-2句话总结以下对话的核心内容：

{conversation_text}

总结："""
            
            # ✅ 详细日志：API调用前
            print(f"    📡 开始调用DeepSeek API...")
            print(f"    📝 Prompt长度: {len(prompt)}字符")
            print(f"    📋 消息数量: {len(messages)}条")
            api_start = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.3
                    },
                    timeout=30.0
                )
                
                api_elapsed = time.time() - api_start
                
                # ✅ 详细日志：API响应
                print(f"    ⏱️  API响应时间: {api_elapsed:.2f}秒")
                print(f"    📊 HTTP状态码: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"    ❌ API错误响应: {response.text[:200]}")
                    raise Exception(f"API错误: {response.status_code}")
                
                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                
                # ✅ 详细日志：生成结果
                print(f"    ✅ 摘要生成成功")
                print(f"    📄 摘要内容: {summary[:80]}...")
                print(f"    💰 Token使用: prompt={result.get('usage', {}).get('prompt_tokens', '?')}, completion={result.get('usage', {}).get('completion_tokens', '?')}")
                
                return summary
        
        except Exception as e:
            print(f"  ⚠️  LLM压缩失败: {e}")
            # 降级到假压缩
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"


# ✅ 继承方式：扩展异步版本支持真实LLM + 详细日志
class MidTermMemoryAsyncWithLLM(MidTermMemoryAsync):
    """扩展MidTermMemoryAsync，添加真实LLM压缩功能 + 详细日志"""
    
    def __init__(
        self, 
        storage, 
        max_turns: int = 10, 
        session_maker=None,
        enable_real_compression: bool = False
    ):
        from src.memory.short_term import ShortTermMemory
        
        # 直接初始化
        self.storage = storage
        self.short_term = ShortTermMemory(max_turns=max_turns)
        self.session_maker = session_maker
        self._compression_tasks = set()
        
        # 添加LLM配置
        self.enable_real_compression = enable_real_compression
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.api_base = "https://api.deepseek.com/v1"
        
        if self.enable_real_compression and not self.api_key:
            print("⚠️  警告：未设置DEEPSEEK_API_KEY，将使用假压缩")
            self.enable_real_compression = False
    
    async def _generate_summary(self, messages: List) -> str:
        """重写：支持真实LLM压缩 + 详细日志"""
        if not self.enable_real_compression:
            # 假压缩
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"
        
        # 真实LLM压缩
        try:
            conversation_text = ""
            for msg in messages:
                role = msg.role if hasattr(msg, 'role') else "user"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                conversation_text += f"{role}: {content}\n"
            
            prompt = f"""请用1-2句话总结以下对话的核心内容：

{conversation_text}

总结："""
            
            # ✅ 详细日志：API调用前
            print(f"    📡 [后台任务] 开始调用DeepSeek API...")
            print(f"    📝 [后台任务] Prompt长度: {len(prompt)}字符")
            print(f"    📋 [后台任务] 消息数量: {len(messages)}条")
            api_start = time.time()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 200,
                        "temperature": 0.3
                    },
                    timeout=30.0
                )
                
                api_elapsed = time.time() - api_start
                
                # ✅ 详细日志：API响应
                print(f"    ⏱️  [后台任务] API响应时间: {api_elapsed:.2f}秒")
                print(f"    📊 [后台任务] HTTP状态码: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"    ❌ [后台任务] API错误响应: {response.text[:200]}")
                    raise Exception(f"API错误: {response.status_code}")
                
                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                
                # ✅ 详细日志：生成结果
                print(f"    ✅ [后台任务] 摘要生成成功")
                print(f"    📄 [后台任务] 摘要内容: {summary[:80]}...")
                print(f"    💰 [后台任务] Token使用: prompt={result.get('usage', {}).get('prompt_tokens', '?')}, completion={result.get('usage', {}).get('completion_tokens', '?')}")
                
                return summary
        
        except Exception as e:
            print(f"  ⚠️  [后台任务] LLM压缩失败: {e}")
            # 降级到假压缩
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"


async def ensure_database():
    """确保测试数据库存在"""
    conn = await asyncpg.connect(
        user=username,
        host='localhost',
        port=5432,
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


class AsyncPerformanceTest:
    """异步vs同步性能对比测试（真实LLM + 详细日志）"""
    
    def __init__(self):
        self.results = {}
        self.async_session_maker = None
        self.engine = None
    
    async def setup(self):
        """初始化测试环境"""
        db_manager = DatabaseManager(DB_URL)
        try:
            await db_manager.drop_tables()
        except:
            pass
        await db_manager.create_tables()
        print("✓ 数据库表已创建\n")
        
        self.engine = create_async_engine(DB_URL, echo=False)
        self.async_session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def teardown(self):
        """清理"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_sync_compression(self, num_messages: int = 100):
        """测试：同步压缩（阻塞用户）+ 真实LLM + 详细日志"""
        print("【测试1】同步压缩（原版，阻塞用户）+ 真实LLM + 详细日志")
        print("=" * 60)
        
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        
        # ✅ 使用继承类，开启真实LLM
        memory = MidTermMemoryWithLLM(
            storage, 
            max_turns=10,
            enable_real_compression=True  # ← 开启真实LLM
        )
        
        try:
            user_id = "sync_user"
            session_id = "sync_session"
            
            print(f"  模拟用户发送 {num_messages} 条消息...")
            print(f"  预期：第69条会阻塞2-6秒（真实LLM压缩）\n")
            
            total_start = time.time()
            message_times = []
            
            for i in range(num_messages):
                msg_start = time.time()
                
                role = "user" if i % 2 == 0 else "assistant"
                await memory.add_message(user_id, session_id, role, f"这是第{i}条测试消息，用于测试压缩性能")
                
                msg_elapsed = time.time() - msg_start
                message_times.append(msg_elapsed)
                
                # 显示慢的消息
                if msg_elapsed > 0.1:  # 超过100ms
                    print(f"    消息{i}: {msg_elapsed*1000:.0f}ms {'🐢 阻塞!' if msg_elapsed > 1 else ''}")
            
            total_elapsed = time.time() - total_start
            
            # 统计
            avg_time = sum(message_times) / len(message_times)
            max_time = max(message_times)
            slow_messages = [t for t in message_times if t > 0.1]
            
            print(f"\n  同步压缩结果:")
            print(f"    总耗时: {total_elapsed:.2f}秒")
            print(f"    平均延迟: {avg_time*1000:.2f}ms")
            print(f"    最大延迟: {max_time*1000:.2f}ms ({max_time:.2f}秒)")
            print(f"    慢消息数: {len(slow_messages)}条 (>100ms)")
            print(f"    吞吐量: {num_messages/total_elapsed:.1f}条/秒")
            
            self.results['sync'] = {
                'total_time': total_elapsed,
                'avg_latency': avg_time * 1000,
                'max_latency': max_time * 1000,
                'slow_count': len(slow_messages),
                'throughput': num_messages / total_elapsed
            }
        
        finally:
            await session.close()
    
    async def test_async_compression(self, num_messages: int = 100):
        """测试：异步压缩（不阻塞用户）+ 真实LLM + 详细日志"""
        print("\n【测试2】异步压缩（新版，不阻塞用户）+ 真实LLM + 详细日志")
        print("=" * 60)
        
        session = self.async_session_maker()
        storage = PostgreSQLStorage(session)
        
        # ✅ 使用继承类，开启真实LLM，传入session_maker
        memory = MidTermMemoryAsyncWithLLM(
            storage, 
            max_turns=10,
            session_maker=self.async_session_maker,  # ← 必须传！
            enable_real_compression=True  # ← 开启真实LLM
        )
        
        try:
            user_id = "async_user"
            session_id = "async_session"
            
            print(f"  模拟用户发送 {num_messages} 条消息...")
            print(f"  预期：所有消息立即返回，LLM压缩在后台\n")
            
            total_start = time.time()
            message_times = []
            
            for i in range(num_messages):
                msg_start = time.time()
                
                role = "user" if i % 2 == 0 else "assistant"
                await memory.add_message(user_id, session_id, role, f"这是第{i}条测试消息，用于测试压缩性能")
                
                msg_elapsed = time.time() - msg_start
                message_times.append(msg_elapsed)
                
                # 显示慢的消息（不应该有）
                if msg_elapsed > 0.1:
                    print(f"    消息{i}: {msg_elapsed*1000:.0f}ms ⚠️ 不应该慢!")
            
            total_elapsed = time.time() - total_start
            
            # 等待后台任务完成
            print(f"\n  ✓ 所有消息已发送（用户无感知延迟）")
            print(f"  ⏳ 等待后台LLM压缩完成...\n")
            await memory.wait_for_compression()
            
            total_with_bg = time.time() - total_start
            
            # 统计
            avg_time = sum(message_times) / len(message_times)
            max_time = max(message_times)
            slow_messages = [t for t in message_times if t > 0.1]
            
            print(f"\n  异步压缩结果:")
            print(f"    总耗时（用户感知）: {total_elapsed:.2f}秒 ← 这是用户体验")
            print(f"    总耗时（含后台）: {total_with_bg:.2f}秒 ← 后台真实耗时")
            print(f"    平均延迟: {avg_time*1000:.2f}ms")
            print(f"    最大延迟: {max_time*1000:.2f}ms")
            print(f"    慢消息数: {len(slow_messages)}条 (>100ms)")
            print(f"    吞吐量: {num_messages/total_elapsed:.1f}条/秒")
            
            self.results['async'] = {
                'total_time': total_elapsed,
                'total_with_bg': total_with_bg,
                'avg_latency': avg_time * 1000,
                'max_latency': max_time * 1000,
                'slow_count': len(slow_messages),
                'throughput': num_messages / total_elapsed
            }
        
        finally:
            await session.close()
    
    def print_comparison(self):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("性能对比总结（真实LLM压缩）")
        print("=" * 60)
        
        if 'sync' not in self.results or 'async' not in self.results:
            print("测试未完成")
            return
        
        sync = self.results['sync']
        async_r = self.results['async']
        
        print("\n【用户体验对比】（发送100条消息）")
        print(f"  同步压缩：")
        print(f"    总耗时: {sync['total_time']:.2f}秒")
        print(f"    最大延迟: {sync['max_latency']:.0f}ms ({sync['max_latency']/1000:.1f}秒)")
        print(f"    慢消息: {sync['slow_count']}条")
        print(f"    用户体验: {'❌ 卡顿明显' if sync['slow_count'] > 0 else '✅ 流畅'}")
        
        print(f"\n  异步压缩：")
        print(f"    总耗时: {async_r['total_time']:.2f}秒")
        print(f"    最大延迟: {async_r['max_latency']:.0f}ms")
        print(f"    慢消息: {async_r['slow_count']}条")
        print(f"    用户体验: {'✅ 完全流畅' if async_r['slow_count'] == 0 else '⚠️ 偶尔卡顿'}")
        
        print(f"\n【性能提升】")
        speedup = sync['total_time'] / async_r['total_time']
        latency_improve = sync['max_latency'] / async_r['max_latency']
        throughput_gain = (async_r['throughput']/sync['throughput']-1)*100
        
        print(f"  用户感知耗时: {speedup:.1f}x 更快")
        print(f"  最大延迟降低: {latency_improve:.0f}x")
        print(f"  吞吐量提升: {throughput_gain:.0f}%")
        
        print(f"\n【LLM压缩开销】")
        if 'total_with_bg' in async_r:
            bg_overhead = async_r['total_with_bg'] - async_r['total_time']
            print(f"  后台压缩耗时: {bg_overhead:.2f}秒（用户无感知）")
            print(f"  压缩次数: 1次（50条触发）")
            print(f"  单次压缩耗时: {bg_overhead:.2f}秒")
        
        print(f"\n【结论】")
        if async_r['slow_count'] == 0 and sync['slow_count'] > 0:
            print(f"  ✅ 异步压缩完全消除了用户感知的LLM延迟")
            print(f"  ✅ 用户体验提升 {speedup:.0f} 倍")
            print(f"  ✅ 强烈推荐在生产环境使用异步压缩")
        else:
            print(f"  ⚠️  测试结果异常，请检查配置")
        
        print("\n" + "=" * 60)


async def main():
    """运行异步vs同步性能对比（真实LLM + 详细日志）"""
    print("=" * 60)
    print("异步压缩 vs 同步压缩 - 性能对比测试（真实LLM + 详细日志）")
    print("=" * 60)
    print(f"数据库: {DB_URL}")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    print(f"LLM API: https://api.deepseek.com")
    print(f"API Key: {'✅ 已配置 (' + api_key[:10] + '...)' if api_key else '❌ 未配置'}\n")
    
    if not api_key:
        print("❌ 请在.env中配置DEEPSEEK_API_KEY")
        return
    
    await ensure_database()
    
    test = AsyncPerformanceTest()
    
    try:
        await test.setup()
        
        # 测试同步压缩（真实LLM）
        await test.test_sync_compression(num_messages=100)
        
        # 测试异步压缩（真实LLM）
        await test.test_async_compression(num_messages=100)
        
        # 打印对比
        test.print_comparison()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await test.teardown()
        print("\n测试完成！")


if __name__ == "__main__":
    asyncio.run(main())