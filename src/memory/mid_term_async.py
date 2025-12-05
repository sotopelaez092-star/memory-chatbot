"""
中期记忆管理器（异步压缩版本）

关键改进：
1. 压缩不阻塞用户请求
2. 后台任务处理压缩
3. 后台任务使用独立session
"""

from typing import List, Dict, Optional
import asyncio
from .short_term import ShortTermMemory
from .postgres_storage import PostgreSQLStorage


class MidTermMemoryAsync:
    """
    中期记忆管理器（异步压缩版本）
    
    核心改进：
    - add_message() 立即返回，不等待压缩
    - 压缩在后台异步执行
    - 用户无感知延迟
    """

    def __init__(
        self,
        storage: PostgreSQLStorage,
        max_turns: int = 10,
        session_maker = None  # ← 新增：用于创建独立session
    ):
        """
        初始化中期记忆管理器
        
        Args:
            storage: PostgreSQL存储实例
            max_turns: 最大轮数，默认10轮
            session_maker: AsyncSession工厂（用于后台任务）
        """
        self.storage = storage
        self.short_term = ShortTermMemory(max_turns=max_turns)
        self.session_maker = session_maker  # 保存session_maker
        
        # 后台任务管理
        self._compression_tasks = set()

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None
    ) -> None:
        """
        添加消息（异步压缩版本）
        
        关键改进：
        1. 添加消息立即返回
        2. 压缩任务在后台执行
        3. 用户无感知延迟
        """
        # 1. 添加到短期记忆
        self.short_term.add_message(role, content)
        
        # 2. 检查是否溢出
        overflow = self.short_term.check_overflow()
        
        # 3. 如果溢出，保存到PostgreSQL
        if overflow:
            # 获取会话ID
            conv = await self.storage.get_or_create_conversation(
                user_id=user_id,
                session_id=session_id
            )
            
            # 保存溢出的消息
            await self.storage.add_messages(conv.id, overflow)
            
            # 4. 检查是否需要压缩
            all_messages = await self.storage.query_messages(conv.id)
            total_count = len(all_messages)
            
            # 每50条触发一次压缩
            if total_count > 0 and total_count % 50 == 0:
                # ✅ 关键改进：不等待压缩，立即返回
                task = asyncio.create_task(
                    self._compress_in_background(
                        conv.id,
                        total_count
                    )
                )
                
                # 跟踪任务（防止被垃圾回收）
                self._compression_tasks.add(task)
                task.add_done_callback(self._compression_tasks.discard)
                
                print(f"✓ 触发异步压缩：当前 {total_count} 条消息（后台处理）")
    
    async def _compress_in_background(
        self,
        conversation_id: int,
        total_count: int
    ) -> None:
        """
        后台压缩任务（使用独立session）
        
        Args:
            conversation_id: 会话ID
            total_count: 当前总消息数
        """
        # ✅ 创建独立的session
        if not self.session_maker:
            print(f"  ⚠️  无法创建独立session，跳过后台压缩")
            return
        
        bg_session = self.session_maker()
        bg_storage = PostgreSQLStorage(bg_session)
        
        try:
            print(f"  🔄 后台压缩开始...")
            
            # 查询最近50条消息
            recent_messages = await bg_storage.query_messages(
                conversation_id=conversation_id,
                limit=50
            )
            
            # 1. 生成摘要
            summary_text = await self._generate_summary(recent_messages)
            
            # 计算时间范围
            start = total_count - 49
            end = total_count
            time_range = f"msg_{start}_to_{end}"
            
            # 保存摘要
            await bg_storage.save_summary(
                conversation_id=conversation_id,
                time_range=time_range,
                summary_text=summary_text
            )
            print(f"  ✓ 后台压缩完成: {time_range}")
            
            # 2. 提取用户画像
            profile_updates = await self._extract_user_profile(recent_messages)
            
            if profile_updates:
                # 更新画像
                await bg_storage.upsert_profile(
                    conversation_id=conversation_id,
                    profile_data=profile_updates
                )
                print(f"  ✓ 画像更新完成: {list(profile_updates.keys())}")
        
        except Exception as e:
            # 失败不影响主流程
            print(f"  ❌ 后台压缩失败: {e}")
        
        finally:
            # ✅ 关闭独立session
            await bg_session.close()

    async def _generate_summary(self, messages: List) -> str:
        """生成摘要（占位符）"""
        user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
        assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
        return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"

    async def _extract_user_profile(self, messages: List) -> Dict[str, str]:
        """提取用户画像（占位符）"""
        return {}
    
    async def wait_for_compression(self) -> None:
        """等待所有后台压缩任务完成"""
        if self._compression_tasks:
            print(f"⏳ 等待 {len(self._compression_tasks)} 个后台压缩任务...")
            await asyncio.gather(*self._compression_tasks, return_exceptions=True)
            print(f"✓ 所有后台任务完成")
    
    async def load_recent_history(
        self,
        user_id: str,
        session_id: str,
        count: int = 10
    ) -> None:
        """从PostgreSQL加载最近的消息到短期记忆"""
        conv = await self.storage.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id
        )
        
        messages = await self.storage.query_messages(
            conversation_id=conv.id,
            limit=count
        )
        
        for msg in reversed(messages):
            self.short_term.add_message(
                role=msg.role,
                content=msg.content
            )
        
        print(f"✓ 加载 {len(messages)} 条消息到短期记忆")

    async def get_context_for_llm(
        self,
        user_id: str,
        session_id: str,
        max_tokens: int = 4096,
        use_compression: bool = True
    ) -> List[Dict]:
        """获取给LLM的上下文"""
        conv = await self.storage.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id
        )
        
        if not use_compression:
            return await self._get_context_sliding_window(conv.id, max_tokens)
        
        context = []
        used_tokens = 0
        
        # 1. 获取用户画像
        profile = await self.storage.get_profile(conv.id)
        if profile:
            profile_text = ", ".join([f"{k}: {v}" for k, v in profile.items()])
            profile_msg = {
                "role": "system",
                "content": f"用户信息：{profile_text}"
            }
            context.append(profile_msg)
            used_tokens += len(profile_text) // 4
        
        # 2. 获取历史摘要
        summaries = await self.storage.get_summaries(conv.id)
        if summaries:
            recent_summaries = summaries[-3:]
            summary_texts = []
            for s in recent_summaries:
                summary_texts.append(f"[{s.time_range}] {s.summary_text}")
            
            summary_msg = {
                "role": "system",
                "content": "历史对话摘要：\n" + "\n".join(summary_texts)
            }
            context.append(summary_msg)
            used_tokens += sum(len(s.summary_text) for s in recent_summaries) // 4
        
        # 3. 获取短期记忆
        short_messages = self.short_term.get_messages()
        context.extend(short_messages)
        used_tokens += len(short_messages) * 50
        
        # 4. 补充历史消息
        remaining_tokens = max_tokens - used_tokens
        if remaining_tokens > 500:
            need_count = remaining_tokens // 50
            history_messages = await self.storage.query_messages(
                conversation_id=conv.id,
                limit=need_count
            )
            
            for msg in reversed(history_messages):
                context.insert(len(context) - len(short_messages), {
                    "role": msg.role,
                    "content": msg.content
                })
        
        return context

    async def _get_context_sliding_window(
        self,
        conversation_id: int,
        max_tokens: int
    ) -> List[Dict]:
        """纯滑动窗口策略"""
        short_messages = self.short_term.get_messages()
        short_tokens = len(short_messages) * 50
        remaining_tokens = max_tokens - short_tokens
        
        if remaining_tokens < 100:
            return short_messages
        
        need_count = remaining_tokens // 50
        history_messages = await self.storage.query_messages(
            conversation_id=conversation_id,
            limit=need_count
        )
        
        context = []
        for msg in reversed(history_messages):
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        context.extend(short_messages)
        return context
    
    def get_short_term_count(self) -> int:
        """获取短期记忆中的消息数量"""
        return len(self.short_term)
    
    async def clear_session(
        self,
        user_id: str,
        session_id: str
    ) -> None:
        """清空会话"""
        self.short_term.clear()
        print(f"✓ 会话已清空: {user_id}:{session_id}")