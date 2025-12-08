"""
真实Agent场景性能测试 - 改进版

场景特点：
1. 大规模历史数据：每用户500条历史消息
2. 高频读取：80%读 + 20%写（模拟Agent工作流）
3. 大批量读取：每次读50条消息（模拟上下文窗口）
4. 并发压力：20个用户同时对话
"""
import asyncio
import time
import random
import statistics
import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.mid_term_async import MidTermMemoryAsync
from src.memory.mid_term_with_redis import MidTermMemoryWithRedis
from src.memory.postgres_storage import PostgreSQLStorage
from src.memory.redis_storage import RedisStorage
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


# ==================== 测试配置 ====================

NUM_USERS = 20              # 并发用户数
HISTORY_MESSAGES = 500      # 每用户历史消息数
TEST_ROUNDS = 50            # 测试轮数
READ_RATIO = 0.8            # 80%操作是读取
CONTEXT_SIZE = 50           # 每次读取50条消息（模拟上下文窗口）


@dataclass
class PerformanceMetrics:
    """性能指标"""
    latencies: List[float]
    read_latencies: List[float]
    write_latencies: List[float]
    total_time: float
    cache_hit_rate: float = 0.0
    
    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0
    
    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]
    
    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]
    
    @property
    def avg(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0
    
    @property
    def throughput(self) -> float:
        return len(self.latencies) / self.total_time if self.total_time > 0 else 0


async def prepare_history(
    session_maker,
    user_id: str,
    num_messages: int,
    use_redis: bool,
    redis_storage=None
) -> None:
    """准备历史消息（模拟长对话历史）"""
    session = session_maker()
    pg_storage = PostgreSQLStorage(session)
    
    if use_redis:
        memory = MidTermMemoryWithRedis(
            pg_storage=pg_storage,
            redis_storage=redis_storage,
            max_turns=5,
            cache_ttl=300
        )
    else:
        memory = MidTermMemoryAsync(
            storage=pg_storage,
            max_turns=5,
            session_maker=session_maker
        )
    
    session_id = f"session_{user_id}"
    
    try:
        # 批量添加历史消息
        for i in range(num_messages):
            await memory.add_message(
                user_id, session_id,
                "user" if i % 2 == 0 else "assistant",
                f"历史消息 {i+1}",
                tokens=50
            )
        
    finally:
        await session.close()


async def simulate_agent_conversation(
    session_maker,
    user_id: str,
    num_rounds: int,
    read_ratio: float,
    context_size: int,
    use_redis: bool,
    redis_storage=None
) -> Dict[str, List[float]]:
    """
    模拟Agent对话工作流
    
    真实Agent流程：
    1. 读取历史上下文（50条消息）
    2. 构造prompt + 调用LLM
    3. 写入新消息
    
    测试中：80%读 + 20%写
    """
    session = session_maker()
    pg_storage = PostgreSQLStorage(session)
    
    if use_redis:
        memory = MidTermMemoryWithRedis(
            pg_storage=pg_storage,
            redis_storage=redis_storage,
            max_turns=5,
            cache_ttl=300
        )
    else:
        memory = MidTermMemoryAsync(
            storage=pg_storage,
            max_turns=5,
            session_maker=session_maker
        )
    
    session_id = f"session_{user_id}"
    write_latencies = []
    read_latencies = []
    
    try:
        for round_num in range(num_rounds):
            operation = "read" if random.random() < read_ratio else "write"
            
            if operation == "read":
                # 读操作：读取大量历史上下文（模拟给LLM的上下文）
                start = time.time()
                
                if use_redis:
                    await memory.query_messages(user_id, session_id, limit=context_size)
                else:
                    conv = await pg_storage.get_or_create_conversation(user_id, session_id)
                    await pg_storage.query_messages(conv.id, limit=context_size)
                
                elapsed = (time.time() - start) * 1000
                read_latencies.append(elapsed)
            
            else:
                # 写操作：添加新消息
                start = time.time()
                
                await memory.add_message(
                    user_id, session_id,
                    "user" if round_num % 2 == 0 else "assistant",
                    f"新消息 round_{round_num}",
                    tokens=50
                )
                
                elapsed = (time.time() - start) * 1000
                write_latencies.append(elapsed)
        
        return {
            "write_latencies": write_latencies,
            "read_latencies": read_latencies,
            "cache_hit_rate": memory.get_cache_hit_rate() if use_redis else 0.0
        }
    
    finally:
        await session.close()


async def test_without_redis() -> PerformanceMetrics:
    """测试：不带Redis缓存"""
    print("\n" + "=" * 60)
    print("【测试1】不带Redis缓存（纯PostgreSQL）")
    print("=" * 60)
    print(f"数据规模：{NUM_USERS}个用户 × {HISTORY_MESSAGES}条历史 = {NUM_USERS * HISTORY_MESSAGES:,}条消息")
    print(f"测试负载：{NUM_USERS}个用户并发 × {TEST_ROUNDS}轮对话")
    print(f"读写比例：{int(READ_RATIO*100)}%读 + {int((1-READ_RATIO)*100)}%写")
    print(f"读取规模：每次读取{CONTEXT_SIZE}条消息（模拟上下文窗口）")
    
    # 创建数据库连接
    engine = create_async_engine(
        "postgresql+asyncpg://FiaShi@localhost/memory_chatbot_test",
        echo=False,
        pool_size=30,
        max_overflow=20
    )
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    try:
        # 阶段1：准备历史数据
        print(f"\n阶段1：准备历史数据...")
        prep_start = time.time()
        
        prep_tasks = []
        for user_idx in range(NUM_USERS):
            user_id = f"user_no_redis_{user_idx}"
            task = asyncio.create_task(
                prepare_history(
                    async_session_maker, user_id,
                    HISTORY_MESSAGES, use_redis=False
                )
            )
            prep_tasks.append(task)
        
        await asyncio.gather(*prep_tasks)
        prep_time = time.time() - prep_start
        print(f"✅ 历史数据准备完成！耗时: {prep_time:.2f}秒")
        
        # 阶段2：测试读写性能
        print(f"\n阶段2：开始性能测试...")
        test_start = time.time()
        
        test_tasks = []
        for user_idx in range(NUM_USERS):
            user_id = f"user_no_redis_{user_idx}"
            task = asyncio.create_task(
                simulate_agent_conversation(
                    async_session_maker, user_id,
                    TEST_ROUNDS, READ_RATIO, CONTEXT_SIZE, use_redis=False
                )
            )
            test_tasks.append(task)
        
        results = await asyncio.gather(*test_tasks)
        test_time = time.time() - test_start
        
        # 汇总结果
        all_write_latencies = []
        all_read_latencies = []
        
        for result in results:
            all_write_latencies.extend(result["write_latencies"])
            all_read_latencies.extend(result["read_latencies"])
        
        all_latencies = all_write_latencies + all_read_latencies
        
        metrics = PerformanceMetrics(
            latencies=all_latencies,
            read_latencies=all_read_latencies,
            write_latencies=all_write_latencies,
            total_time=test_time,
            cache_hit_rate=0.0
        )
        
        print(f"✅ 测试完成！总耗时: {test_time:.2f}秒")
        return metrics
        
    finally:
        await engine.dispose()


async def test_with_redis() -> PerformanceMetrics:
    """测试：带Redis缓存"""
    print("\n" + "=" * 60)
    print("【测试2】带Redis缓存")
    print("=" * 60)
    print(f"数据规模：{NUM_USERS}个用户 × {HISTORY_MESSAGES}条历史 = {NUM_USERS * HISTORY_MESSAGES:,}条消息")
    print(f"测试负载：{NUM_USERS}个用户并发 × {TEST_ROUNDS}轮对话")
    print(f"读写比例：{int(READ_RATIO*100)}%读 + {int((1-READ_RATIO)*100)}%写")
    print(f"读取规模：每次读取{CONTEXT_SIZE}条消息（模拟上下文窗口）")
    
    # 创建数据库连接
    engine = create_async_engine(
        "postgresql+asyncpg://FiaShi@localhost/memory_chatbot_test",
        echo=False,
        pool_size=30,
        max_overflow=20
    )
    
    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    # 创建Redis连接
    redis_storage = RedisStorage()
    await redis_storage.connect()
    
    try:
        # 阶段1：准备历史数据
        print(f"\n阶段1：准备历史数据...")
        prep_start = time.time()
        
        prep_tasks = []
        for user_idx in range(NUM_USERS):
            user_id = f"user_with_redis_{user_idx}"
            task = asyncio.create_task(
                prepare_history(
                    async_session_maker, user_id,
                    HISTORY_MESSAGES, use_redis=True,
                    redis_storage=redis_storage
                )
            )
            prep_tasks.append(task)
        
        await asyncio.gather(*prep_tasks)
        prep_time = time.time() - prep_start
        print(f"✅ 历史数据准备完成！耗时: {prep_time:.2f}秒")
        
        # 阶段2：测试读写性能
        print(f"\n阶段2：开始性能测试...")
        test_start = time.time()
        
        test_tasks = []
        for user_idx in range(NUM_USERS):
            user_id = f"user_with_redis_{user_idx}"
            task = asyncio.create_task(
                simulate_agent_conversation(
                    async_session_maker, user_id,
                    TEST_ROUNDS, READ_RATIO, CONTEXT_SIZE, use_redis=True,
                    redis_storage=redis_storage
                )
            )
            test_tasks.append(task)
        
        results = await asyncio.gather(*test_tasks)
        test_time = time.time() - test_start
        
        # 汇总结果
        all_write_latencies = []
        all_read_latencies = []
        cache_hit_rates = []
        
        for result in results:
            all_write_latencies.extend(result["write_latencies"])
            all_read_latencies.extend(result["read_latencies"])
            cache_hit_rates.append(result["cache_hit_rate"])
        
        all_latencies = all_write_latencies + all_read_latencies
        avg_cache_hit_rate = sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0
        
        metrics = PerformanceMetrics(
            latencies=all_latencies,
            read_latencies=all_read_latencies,
            write_latencies=all_write_latencies,
            total_time=test_time,
            cache_hit_rate=avg_cache_hit_rate
        )
        
        print(f"✅ 测试完成！总耗时: {test_time:.2f}秒")
        
        # 清理测试数据
        for user_idx in range(NUM_USERS):
            user_id = f"user_with_redis_{user_idx}"
            session_id = f"session_{user_id}"
            await redis_storage.redis.delete(
                redis_storage._message_list_key(user_id, session_id)
            )
        
        return metrics
        
    finally:
        await redis_storage.close()
        await engine.dispose()


def print_metrics(name: str, metrics: PerformanceMetrics):
    """打印性能指标"""
    print(f"\n📊 {name} 性能报告")
    print("=" * 60)
    
    print(f"\n【延迟分布】")
    print(f"  P50 (中位数): {metrics.p50:.2f}ms")
    print(f"  P95 (95分位): {metrics.p95:.2f}ms")
    print(f"  P99 (99分位): {metrics.p99:.2f}ms")
    print(f"  平均延迟: {metrics.avg:.2f}ms")
    
    print(f"\n【读写性能】")
    if metrics.read_latencies:
        read_avg = sum(metrics.read_latencies) / len(metrics.read_latencies)
        read_p95 = sorted(metrics.read_latencies)[int(len(metrics.read_latencies) * 0.95)]
        print(f"  读操作平均: {read_avg:.2f}ms")
        print(f"  读操作P95: {read_p95:.2f}ms")
        print(f"  读操作次数: {len(metrics.read_latencies)}次")
    if metrics.write_latencies:
        write_avg = sum(metrics.write_latencies) / len(metrics.write_latencies)
        print(f"  写操作平均: {write_avg:.2f}ms")
        print(f"  写操作次数: {len(metrics.write_latencies)}次")
    
    print(f"\n【整体性能】")
    print(f"  总请求数: {len(metrics.latencies)}个")
    print(f"  总耗时: {metrics.total_time:.2f}秒")
    print(f"  吞吐量: {metrics.throughput:.2f}请求/秒")
    
    if metrics.cache_hit_rate > 0:
        print(f"\n【缓存效果】")
        print(f"  缓存命中率: {metrics.cache_hit_rate:.2%}")


async def main():
    """主测试流程"""
    print("=" * 60)
    print("🚀 真实Agent场景性能测试 - 改进版")
    print("=" * 60)
    print(f"\n测试配置：")
    print(f"  并发用户: {NUM_USERS}个")
    print(f"  历史消息: 每用户{HISTORY_MESSAGES}条")
    print(f"  测试轮数: 每用户{TEST_ROUNDS}轮")
    print(f"  总交互: {NUM_USERS * TEST_ROUNDS}次")
    print(f"  读写比例: {int(READ_RATIO*100)}%读 / {int((1-READ_RATIO)*100)}%写")
    print(f"  上下文窗口: {CONTEXT_SIZE}条消息")
    
    # 测试1：不带Redis
    metrics_no_redis = await test_without_redis()
    
    # 等待1秒
    await asyncio.sleep(1)
    
    # 测试2：带Redis
    metrics_with_redis = await test_with_redis()
    
    # 打印详细报告
    print_metrics("不带Redis", metrics_no_redis)
    print_metrics("带Redis", metrics_with_redis)
    
    # 对比分析
    print("\n" + "=" * 60)
    print("🎯 性能对比分析")
    print("=" * 60)
    
    print(f"\n【关键指标对比】")
    
    # P50对比
    print(f"\nP50延迟（中位数）：")
    print(f"  不带Redis: {metrics_no_redis.p50:.2f}ms")
    print(f"  带Redis:   {metrics_with_redis.p50:.2f}ms")
    p50_improvement = metrics_no_redis.p50 / metrics_with_redis.p50 if metrics_with_redis.p50 > 0 else 0
    if p50_improvement > 1:
        print(f"  ✅ 提升: {p50_improvement:.2f}x")
    else:
        print(f"  ⚠️ 变化: {p50_improvement:.2f}x")
    
    # P95对比
    print(f"\nP95延迟（95分位）：")
    print(f"  不带Redis: {metrics_no_redis.p95:.2f}ms")
    print(f"  带Redis:   {metrics_with_redis.p95:.2f}ms")
    p95_improvement = metrics_no_redis.p95 / metrics_with_redis.p95 if metrics_with_redis.p95 > 0 else 0
    if p95_improvement > 1:
        print(f"  ✅ 提升: {p95_improvement:.2f}x")
    else:
        print(f"  ⚠️ 变化: {p95_improvement:.2f}x")
    
    # 读操作对比（最关键）
    if metrics_with_redis.read_latencies and metrics_no_redis.read_latencies:
        read_avg_no_redis = sum(metrics_no_redis.read_latencies) / len(metrics_no_redis.read_latencies)
        read_avg_with_redis = sum(metrics_with_redis.read_latencies) / len(metrics_with_redis.read_latencies)
        read_p95_no_redis = sorted(metrics_no_redis.read_latencies)[int(len(metrics_no_redis.read_latencies) * 0.95)]
        read_p95_with_redis = sorted(metrics_with_redis.read_latencies)[int(len(metrics_with_redis.read_latencies) * 0.95)]
        
        read_improvement = read_avg_no_redis / read_avg_with_redis if read_avg_with_redis > 0 else 0
        read_p95_improvement = read_p95_no_redis / read_p95_with_redis if read_p95_with_redis > 0 else 0
        
        print(f"\n📖 读操作性能（核心指标）：")
        print(f"  平均延迟：")
        print(f"    不带Redis: {read_avg_no_redis:.2f}ms")
        print(f"    带Redis:   {read_avg_with_redis:.2f}ms")
        if read_improvement > 1:
            print(f"    ✅ 提升: {read_improvement:.2f}x")
        else:
            print(f"    ⚠️ 变化: {read_improvement:.2f}x")
        
        print(f"  P95延迟：")
        print(f"    不带Redis: {read_p95_no_redis:.2f}ms")
        print(f"    带Redis:   {read_p95_with_redis:.2f}ms")
        if read_p95_improvement > 1:
            print(f"    ✅ 提升: {read_p95_improvement:.2f}x")
        else:
            print(f"    ⚠️ 变化: {read_p95_improvement:.2f}x")
    
    # 吞吐量对比
    print(f"\n吞吐量：")
    print(f"  不带Redis: {metrics_no_redis.throughput:.2f}请求/秒")
    print(f"  带Redis:   {metrics_with_redis.throughput:.2f}请求/秒")
    throughput_improvement = metrics_with_redis.throughput / metrics_no_redis.throughput if metrics_no_redis.throughput > 0 else 0
    if throughput_improvement > 1:
        print(f"  ✅ 提升: {throughput_improvement:.2f}x")
    else:
        print(f"  ⚠️ 变化: {throughput_improvement:.2f}x")
    
    # 总耗时对比
    print(f"\n总耗时：")
    print(f"  不带Redis: {metrics_no_redis.total_time:.2f}秒")
    print(f"  带Redis:   {metrics_with_redis.total_time:.2f}秒")
    time_saved = metrics_no_redis.total_time - metrics_with_redis.total_time
    time_saved_pct = (time_saved / metrics_no_redis.total_time) * 100 if metrics_no_redis.total_time > 0 else 0
    if time_saved > 0:
        print(f"  ✅ 节省: {time_saved:.2f}秒 ({time_saved_pct:.1f}%)")
    else:
        print(f"  ⚠️ 增加: {abs(time_saved):.2f}秒 ({abs(time_saved_pct):.1f}%)")
    
    # 缓存效果
    print(f"\n缓存效果：")
    print(f"  缓存命中率: {metrics_with_redis.cache_hit_rate:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ 真实场景性能测试完成！")
    print("=" * 60)
    
    # 总结
    print(f"\n💡 关键发现：")
    print(f"  1. 数据规模: {NUM_USERS * HISTORY_MESSAGES:,}条历史消息")
    print(f"  2. 读操作占比: {int(READ_RATIO*100)}% (每次读取{CONTEXT_SIZE}条)")
    if read_improvement > 1:
        print(f"  3. 读操作性能提升: {read_improvement:.1f}倍")
    print(f"  4. 缓存命中率: {metrics_with_redis.cache_hit_rate:.1%}")
    if p95_improvement > 1:
        print(f"  5. P95延迟降低: {p95_improvement:.1f}倍")


if __name__ == "__main__":
    asyncio.run(main())