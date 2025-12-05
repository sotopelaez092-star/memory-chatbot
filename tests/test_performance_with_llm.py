"""
性能测试 - 真实LLM压缩（完整修复版）
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
import time
from typing import List, Dict
from dotenv import load_dotenv
import httpx
import json

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.database import DatabaseManager
from src.memory.mid_term import MidTermMemory


# 加载环境变量
load_dotenv()

# 数据库配置
username = os.getenv("USER")
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "memory_chatbot_test"

DB_URL = f"postgresql+asyncpg://{username}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class MidTermMemoryWithLLM(MidTermMemory):
    """
    扩展MidTermMemory，添加真实LLM压缩功能
    
    通过继承扩展，不修改原始代码
    """
    
    def __init__(
        self,
        storage,
        max_turns: int = 10,
        enable_real_compression: bool = False
    ):
        """初始化"""
        from src.memory.short_term import ShortTermMemory
        
        # 直接初始化，不调用父类__init__（避免bug）
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
        """重写父类方法，支持真实LLM压缩"""
        if not self.enable_real_compression:
            # 假压缩（不调用LLM）
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"
        
        # 真实LLM压缩
        try:
            return await self._generate_real_summary(messages)
        except Exception as e:
            print(f"❌ LLM压缩失败: {e}")
            # 降级到假压缩
            user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
            assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
            return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"
    
    async def _generate_real_summary(self, messages: List) -> str:
        """使用DeepSeek生成真实摘要"""
        conversation_text = ""
        for msg in messages:
            role = msg.role if hasattr(msg, 'role') else "user"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            conversation_text += f"{role}: {content}\n"
        
        prompt = f"""请用1-2句话总结以下对话的核心内容：

{conversation_text}

总结："""
        
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
            
            if response.status_code != 200:
                raise Exception(f"API错误: {response.status_code} {response.text}")
            
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            return summary
    
    async def _extract_user_profile(self, messages: List) -> Dict[str, str]:
        """重写父类方法，支持真实LLM提取"""
        if not self.enable_real_compression:
            return {}
        
        try:
            return await self._extract_real_profile(messages)
        except Exception as e:
            print(f"❌ LLM画像提取失败: {e}")
            return {}
    
    async def _extract_real_profile(self, messages: List) -> Dict[str, str]:
        """使用DeepSeek提取真实用户画像"""
        user_messages = [m for m in messages if hasattr(m, 'role') and m.role == "user"]
        
        if not user_messages:
            return {}
        
        user_text = "\n".join([
            m.content if hasattr(m, 'content') else str(m) 
            for m in user_messages
        ])
        
        prompt = f"""分析以下用户的对话，提取用户画像信息。

用户消息：
{user_text}

请以JSON格式返回用户画像，包括但不限于：name, age, location, interests, occupation等。
如果某个信息不确定，不要包含在JSON中。只返回JSON，不要其他文字。

示例：
{{"name": "Tom", "age": "28", "location": "上海", "interests": "编程,AI"}}

JSON："""
        
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
                    "max_tokens": 300,
                    "temperature": 0.1
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"API错误: {response.status_code}")
            
            result = response.json()
            profile_text = result["choices"][0]["message"]["content"].strip()
            
            try:
                # 移除可能的markdown代码块
                if "```json" in profile_text:
                    profile_text = profile_text.split("```json")[1].split("```")[0].strip()
                elif "```" in profile_text:
                    profile_text = profile_text.split("```")[1].split("```")[0].strip()
                
                profile = json.loads(profile_text)
                
                # ✅ 关键修复：转换所有值为字符串
                profile_str = {k: str(v) for k, v in profile.items()}
                
                return profile_str
            except json.JSONDecodeError:
                print(f"⚠️  无法解析画像JSON: {profile_text}")
                return {}


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


class LLMPerformanceTest:
    """LLM性能测试（完整修复版）"""
    
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
        print("✓ 数据库表已创建")
        
        self.engine = create_async_engine(DB_URL, echo=False)
        self.async_session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def teardown(self):
        """清理"""
        if self.engine:
            await self.engine.dispose()
    
    async def test_fake_vs_real_compression(self):
        """
        测试1：假压缩 vs 真实LLM压缩（完整修复版）
        
        关键修复：
        1. 第一阶段：添加70条 → PostgreSQL有50条，触发第1次压缩
        2. 第二阶段：添加48条 → PostgreSQL有98条
        3. 第三阶段：添加2条 → PostgreSQL有100条，触发第2次压缩（测量这次）
        """
        print(f"\n【测试1】假压缩 vs 真实LLM压缩（完整修复版）")
        print("=" * 60)
        
        # === 测试假压缩 ===
        print("\n【假压缩】不调用LLM")
        print("-" * 50)
        
        session1 = self.async_session_maker()
        storage1 = PostgreSQLStorage(session1)
        memory_fake = MidTermMemoryWithLLM(
            storage1, 
            max_turns=10,
            enable_real_compression=False
        )
        
        try:
            user_id = "fake_user"
            session_id = "fake_session"
            
            print(f"  阶段1：添加70条消息（触发第1次压缩）")
            
            for i in range(70):
                role = "user" if i % 2 == 0 else "assistant"
                await memory_fake.add_message(user_id, session_id, role, f"消息{i}")
            
            conv = await storage1.get_or_create_conversation(user_id, session_id)
            db_count = len(await storage1.query_messages(conv.id))
            summaries = await storage1.get_summaries(conv.id)
            
            print(f"    ✓ PostgreSQL: {db_count}条")
            print(f"    ✓ 摘要数量: {len(summaries)}个")
            
            # 阶段2：添加48条到98条
            print(f"\n  阶段2：添加48条到98条")
            for i in range(48):
                role = "user" if i % 2 == 0 else "assistant"
                await memory_fake.add_message(user_id, session_id, role, f"追加{i}")
            
            db_count = len(await storage1.query_messages(conv.id))
            print(f"    ✓ PostgreSQL: {db_count}条")
            
            # 阶段3：测量第99-100条（触发第2次压缩）
            print(f"\n  阶段3：测量第99-100条（触发第2次压缩）")
            start = time.time()
            await memory_fake.add_message(user_id, session_id, "user", "第99条")
            await memory_fake.add_message(user_id, session_id, "assistant", "第100条")
            fake_time = time.time() - start
            
            print(f"    ✓ 假压缩延迟: {fake_time*1000:.2f} ms")
            
            self.results['fake_compression'] = {
                'latency': fake_time * 1000
            }
        
        finally:
            await session1.close()
        
        # === 测试真实LLM压缩 ===
        print("\n【真实LLM压缩】调用DeepSeek API")
        print("-" * 50)
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("  ❌ 未设置DEEPSEEK_API_KEY，跳过真实压缩测试")
            self.results['real_compression'] = {
                'latency': 0,
                'skipped': True
            }
            return
        
        print(f"  ✓ API Key已设置: {api_key[:10]}...")
        
        session2 = self.async_session_maker()
        storage2 = PostgreSQLStorage(session2)
        memory_real = MidTermMemoryWithLLM(
            storage2,
            max_turns=10,
            enable_real_compression=True
        )
        
        try:
            user_id = "real_user"
            session_id = "real_session"
            
            print(f"  阶段1：添加70条消息（触发第1次压缩）")
            
            for i in range(70):
                role = "user" if i % 2 == 0 else "assistant"
                content = f"我是Tom，我今年28岁，在上海工作。这是第{i}条消息。"
                await memory_real.add_message(user_id, session_id, role, content)
            
            conv = await storage2.get_or_create_conversation(user_id, session_id)
            db_count = len(await storage2.query_messages(conv.id))
            summaries = await storage2.get_summaries(conv.id)
            profiles = await storage2.get_profile(conv.id)
            
            print(f"    ✓ PostgreSQL: {db_count}条")
            print(f"    ✓ 摘要数量: {len(summaries)}个")
            if summaries:
                print(f"    ✓ 摘要内容: {summaries[0].summary_text[:60]}...")
            if profiles:
                print(f"    ✓ 用户画像: {profiles}")
            
            # 阶段2：添加48条到98条
            print(f"\n  阶段2：添加48条到98条")
            for i in range(48):
                role = "user" if i % 2 == 0 else "assistant"
                await memory_real.add_message(user_id, session_id, role, f"追加{i}")
            
            db_count = len(await storage2.query_messages(conv.id))
            print(f"    ✓ PostgreSQL: {db_count}条")
            
            # 阶段3：测量第99-100条（触发第2次LLM压缩）
            print(f"\n  阶段3：测量第99-100条（触发第2次LLM压缩）")
            print(f"    ⏱️  预计耗时: 5-6秒...")
            
            start = time.time()
            await memory_real.add_message(user_id, session_id, "user", "我喜欢编程和AI")
            await memory_real.add_message(user_id, session_id, "assistant", "很高兴知道你的兴趣")
            real_time = time.time() - start
            
            print(f"    ✓ 真实LLM压缩延迟: {real_time*1000:.2f} ms ({real_time:.2f}秒)")
            
            self.results['real_compression'] = {
                'latency': real_time * 1000,
                'skipped': False
            }
        
        finally:
            await session2.close()
    
    async def test_compression_batch(self, count: int = 3):
        """
        测试2：批量压缩测试（完整修复版）
        
        每个batch独立测试，确保触发LLM压缩
        """
        print(f"\n【测试2】批量压缩测试（{count}次，独立实例）")
        print("=" * 60)
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("  ❌ 未设置DEEPSEEK_API_KEY，跳过")
            return
        
        compression_times = []
        
        for batch in range(count):
            print(f"\n第{batch+1}次测试...")
            
            # 创建独立的session和memory
            session = self.async_session_maker()
            storage = PostgreSQLStorage(session)
            memory = MidTermMemoryWithLLM(
                storage,
                max_turns=10,
                enable_real_compression=True
            )
            
            try:
                user_id = f"batch_user_{batch}"
                session_id = f"batch_session_{batch}"
                
                # 阶段1：添加70条（触发第1次压缩）
                print(f"  阶段1：添加70条...")
                for i in range(70):
                    role = "user" if i % 2 == 0 else "assistant"
                    await memory.add_message(user_id, session_id, role, f"消息{i}")
                
                conv = await storage.get_or_create_conversation(user_id, session_id)
                summaries = await storage.get_summaries(conv.id)
                print(f"    ✓ 摘要数量: {len(summaries)}个")
                
                if not summaries:
                    print(f"  ⚠️  第1次压缩未触发（跳过）")
                    continue
                
                # 阶段2：添加28条到78条
                print(f"  阶段2：添加28条到78条...")
                for i in range(48):
                    role = "user" if i % 2 == 0 else "assistant"
                    await memory.add_message(user_id, session_id, role, f"追加{i}")
                
                db_count = len(await storage.query_messages(conv.id))
                print(f"    ✓ PostgreSQL: {db_count}条")
                
                # 阶段3：测量第79-80条（触发第2次LLM压缩）
                print(f"  阶段3：测量第79-80条（触发LLM压缩）...")
                start = time.time()
                await memory.add_message(user_id, session_id, "user", "第79条")
                await memory.add_message(user_id, session_id, "assistant", "第80条")
                elapsed = time.time() - start
                
                compression_times.append(elapsed)
                print(f"    ✓ 延迟: {elapsed*1000:.2f} ms ({elapsed:.2f}秒)")
            
            finally:
                await session.close()
        
        if compression_times:
            avg_time = sum(compression_times) / len(compression_times)
            max_time = max(compression_times)
            min_time = min(compression_times)
            
            print(f"\n批量测试结果（{len(compression_times)}次有效）:")
            print(f"  平均延迟: {avg_time*1000:.2f} ms ({avg_time:.2f}秒)")
            print(f"  最大延迟: {max_time*1000:.2f} ms ({max_time:.2f}秒)")
            print(f"  最小延迟: {min_time*1000:.2f} ms ({min_time:.2f}秒)")
            
            self.results['batch_compression'] = {
                'avg_latency': avg_time * 1000,
                'max_latency': max_time * 1000,
                'min_latency': min_time * 1000,
                'valid_count': len(compression_times)
            }
        else:
            print("\n  ❌ 没有有效的压缩测试")
    
    def print_summary(self):
        """打印性能摘要"""
        print("\n" + "=" * 60)
        print("LLM性能测试摘要")
        print("=" * 60)
        
        if 'fake_compression' not in self.results or 'real_compression' not in self.results:
            print("\n测试未完成")
            return
        
        print("\n【对比结果】")
        fake = self.results['fake_compression']
        real = self.results['real_compression']
        
        print(f"  假压缩延迟:      {fake['latency']:8.2f} ms")
        
        if not real.get('skipped'):
            print(f"  真实LLM压缩延迟: {real['latency']:8.2f} ms ({real['latency']/1000:.2f}秒)")
            
            if fake['latency'] > 0:
                ratio = real['latency'] / fake['latency']
                print(f"  差距:            {real['latency'] - fake['latency']:8.2f} ms ({ratio:.0f}x)")
            
            if 'batch_compression' in self.results:
                batch = self.results['batch_compression']
                print(f"\n【批量测试】（{batch['valid_count']}次有效）")
                print(f"  平均延迟: {batch['avg_latency']:.2f} ms ({batch['avg_latency']/1000:.2f}秒)")
                print(f"  最大延迟: {batch['max_latency']:.2f} ms ({batch['max_latency']/1000:.2f}秒)")
                print(f"  最小延迟: {batch['min_latency']:.2f} ms ({batch['min_latency']/1000:.2f}秒)")
            
            print("\n【影响分析】")
            avg_latency = real['latency']
            if 'batch_compression' in self.results:
                avg_latency = self.results['batch_compression']['avg_latency']
            
            if avg_latency > 2000:
                print(f"  ❌ 压缩延迟 >{avg_latency/1000:.1f}秒，严重影响用户体验")
                print(f"  💡 建议：必须使用异步压缩！")
            elif avg_latency > 1000:
                print(f"  ⚠️  压缩延迟 >{avg_latency/1000:.1f}秒，影响用户体验")
                print(f"  💡 建议：强烈建议使用异步压缩")
            elif avg_latency > 500:
                print(f"  ⚠️  压缩延迟 >{avg_latency:.0f}ms，有感知延迟")
                print(f"  💡 建议：考虑使用异步压缩")
            else:
                print(f"  ✅ 压缩延迟 <500ms，可接受")
        else:
            print("  ⚠️  真实LLM测试已跳过")
        
        print("\n" + "=" * 60)


async def main():
    """运行LLM性能测试"""
    print("=" * 60)
    print("LLM压缩性能测试（完整修复版）")
    print("=" * 60)
    print(f"数据库: {DB_URL}")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("\n⚠️  警告：未设置DEEPSEEK_API_KEY")
        print("真实LLM测试将被跳过")
        print("\n设置方法:")
        print("  1. 在项目根目录创建 .env 文件")
        print("  2. 添加：DEEPSEEK_API_KEY=your-key")
        print()
    else:
        print(f"✓ DeepSeek API Key已设置: {api_key[:10]}...")
    
    await ensure_database()
    
    test = LLMPerformanceTest()
    
    try:
        print("\n初始化测试环境...")
        await test.setup()
        
        # 运行测试
        await test.test_fake_vs_real_compression()
        
        if api_key:
            await test.test_compression_batch(count=3)
        
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