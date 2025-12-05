"""
中期记忆管理器（同步版本）

整合短期记忆 (Deque) 和中期存储 (PostgreSQL)
"""

from typing import List, Dict, Optional
from .short_term import ShortTermMemory
from .postgres_storage import PostgreSQLStorage


class MidTermMemory:
    """
    中期记忆管理器（同步压缩版本）
    
    职责：
    1. 管理短期记忆（Deque，最近10轮）
    2. 溢出时保存到PostgreSQL
    3. 定期压缩（每50条生成摘要+画像）
    4. 提供给LLM的上下文（支持压缩和非压缩模式）
    """

    def __init__(
        self,
        storage: PostgreSQLStorage,
        max_turns: int = 10
    ):
        """
        初始化中期记忆管理器
        
        Args:
            storage: PostgreSQL存储实例
            max_turns: 最大轮数，默认10轮
        """
        self.storage = storage
        self.short_term = ShortTermMemory(max_turns=max_turns)

    async def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None
    ) -> None:
        """
        添加消息（核心方法 + 压缩触发）
        
        流程:
        1. 添加到短期记忆
        2. 检查是否溢出
        3. 如果溢出，保存到PostgreSQL
        4. 检查是否达到50条，触发压缩
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            role: 角色（user/assistant/system）
            content: 消息内容
            tokens: Token数（可选）
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
                print(f"✓ 触发压缩：当前 {total_count} 条消息")
                await self._compress_recent_messages(
                    conv.id,
                    all_messages[-50:],
                    total_count
                )
    
    async def _compress_recent_messages(
        self,
        conversation_id: int,
        recent_messages: List,
        total_count: int
    ) -> None:
        """
        压缩最近的消息（生成摘要 + 提取画像）
        
        Args:
            conversation_id: 会话ID
            recent_messages: 最近的消息列表（Message对象）
            total_count: 当前总消息数
        """
        # 1. 生成摘要
        summary_text = await self._generate_summary(recent_messages)
        
        # 计算时间范围
        start = total_count - 49
        end = total_count
        time_range = f"msg_{start}_to_{end}"
        
        # 保存摘要
        await self.storage.save_summary(
            conversation_id=conversation_id,
            time_range=time_range,
            summary_text=summary_text
        )
        print(f"  ✓ 生成摘要: {time_range}")
        
        # 2. 提取用户画像
        profile_updates = await self._extract_user_profile(recent_messages)
        
        if profile_updates:
            # 更新画像
            await self.storage.upsert_profile(
                conversation_id=conversation_id,
                profile_data=profile_updates
            )
            print(f"  ✓ 更新画像: {list(profile_updates.keys())}")

    async def _generate_summary(self, messages: List) -> str:
        """
        生成摘要（占位符 - 不调用LLM）
        
        Args:
            messages: 消息列表
        
        Returns:
            摘要文本
        """
        user_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "user")
        assistant_count = sum(1 for m in messages if hasattr(m, 'role') and m.role == "assistant")
        return f"对话包含{len(messages)}条消息，用户发送了{user_count}条，助手回复了{assistant_count}条。"

    async def _extract_user_profile(self, messages: List) -> Dict[str, str]:
        """
        提取用户画像（占位符 - 简单关键词匹配）
        
        Args:
            messages: 消息列表
        
        Returns:
            用户画像字典
        """
        # 简单的关键词匹配（占位符）
        return {}

    async def load_recent_history(
        self,
        user_id: str,
        session_id: str,
        count: int = 10
    ) -> None:
        """
        从PostgreSQL加载最近的消息到短期记忆
        
        使用场景：
        1. 程序重启后
        2. 用户重新上线
        3. 需要恢复上下文
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            count: 加载消息数量（默认10条，即5轮）
        """
        # 1. 获取会话
        conv = await self.storage.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id
        )
        
        # 2. 从PostgreSQL查询最近count条消息（返回倒序）
        messages = await self.storage.query_messages(
            conversation_id=conv.id,
            limit=count
        )
        
        # 3. 反转后添加到短期记忆（保证正序）
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
        """
        获取给LLM的上下文
        
        策略：
        - use_compression=True: 使用画像+摘要+最近消息（推荐）
        - use_compression=False: 纯滑动窗口（基础版）
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            max_tokens: 最大token数（默认4096）
            use_compression: 是否使用压缩信息
        
        Returns:
            消息列表 [{"role": "user", "content": "..."}, ...]
        """
        # 获取会话
        conv = await self.storage.get_or_create_conversation(
            user_id=user_id,
            session_id=session_id
        )
        
        # 如果不使用压缩，走基础版本（纯滑动窗口）
        if not use_compression:
            return await self._get_context_sliding_window(conv.id, max_tokens)
        
        # 使用压缩信息
        context = []
        used_tokens = 0
        
        # 1. 获取用户画像（约200 tokens）
        profile = await self.storage.get_profile(conv.id)
        if profile:
            profile_text = ", ".join([f"{k}: {v}" for k, v in profile.items()])
            profile_msg = {
                "role": "system",
                "content": f"用户信息：{profile_text}"
            }
            context.append(profile_msg)
            used_tokens += len(profile_text) // 4
        
        # 2. 获取历史摘要（约500 tokens）
        summaries = await self.storage.get_summaries(conv.id)
        if summaries:
            # 只取最近3个摘要（避免太长）
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
        
        # 3. 获取短期记忆（完整，约1000 tokens）
        short_messages = self.short_term.get_messages()
        context.extend(short_messages)
        used_tokens += len(short_messages) * 50
        
        # 4. 如果还有余量，补充一些历史原始消息
        remaining_tokens = max_tokens - used_tokens
        if remaining_tokens > 500:  # 至少剩500 tokens才补充
            need_count = remaining_tokens // 50
            history_messages = await self.storage.query_messages(
                conversation_id=conv.id,
                limit=need_count
            )
            
            # 插入到短期记忆之前
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
        """
        纯滑动窗口策略（不使用压缩信息）
        
        这是之前实现的基础版本
        """
        # 1. 获取短期记忆
        short_messages = self.short_term.get_messages()
        
        # 2. 估算短期的tokens
        short_tokens = len(short_messages) * 50
        
        # 3. 计算剩余tokens
        remaining_tokens = max_tokens - short_tokens
        
        # 4. 如果不够100 tokens，只返回短期
        if remaining_tokens < 100:
            return short_messages
        
        # 5. 计算需要多少条历史
        need_count = remaining_tokens // 50
        
        # 6. 查询历史
        history_messages = await self.storage.query_messages(
            conversation_id=conversation_id,
            limit=need_count
        )
        
        # 7. 组合
        context = []
        
        # 添加历史（注意反转）
        for msg in reversed(history_messages):
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # 添加短期
        context.extend(short_messages)
        
        return context
    
    def get_short_term_count(self) -> int:
        """
        获取短期记忆中的消息数量
        
        Returns:
            消息数量
        """
        return len(self.short_term)
    
    async def clear_session(
        self,
        user_id: str,
        session_id: str
    ) -> None:
        """
        清空会话（清空短期记忆）
        
        注意：只清空短期记忆，PostgreSQL中的历史保留
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
        """
        self.short_term.clear()
        print(f"✓ 会话已清空: {user_id}:{session_id}")