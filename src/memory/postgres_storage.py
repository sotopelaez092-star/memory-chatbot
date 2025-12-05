"""
PostgreSQL存储层（异步）

提供13个方法操作数据库
"""

from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy import select, delete, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Conversation, Message, UserProfile, Summary
from sqlalchemy.dialects.postgresql import insert as pg_insert


class PostgreSQLStorage:
    """PostgreSQL异步存储管理器"""
    
    def __init__(self, session: AsyncSession):
        """
        初始化
        
        Args:
            session: SQLAlchemy异步会话
        """
        self.session = session
    
    # ========== conversations表方法 ==========
    
    async def get_or_create_conversation(
        self,
        user_id: str,
        session_id: str,
        title: Optional[str] = None
    ) -> Conversation:
        """
        获取或创建会话
        
        Args:
            user_id: 用户ID
            session_id: 会话ID
            title: 会话标题（可选）
        
        Returns:
            Conversation对象
        """
        # 1. 先尝试查询
        stmt = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.session_id == session_id
        )
        result = await self.session.execute(stmt)
        conversation = result.scalar_one_or_none()
        
        # 2. 如果不存在，创建新的
        if conversation is None:
            conversation = Conversation(
                user_id=user_id,
                session_id=session_id,
                title=title
            )
            self.session.add(conversation)
            await self.session.commit()
            await self.session.refresh(conversation)  # 刷新获取数据库生成的id
        
        return conversation
    
    async def delete_conversation(self, conversation_id: int) -> bool:
        """
        删除会话（及其所有关联数据）
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            是否删除成功
        """

        stmt = delete(Conversation).where(Conversation.id == conversation_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def update_conversation_time(self, conversation_id: int) -> bool:
        """
        更新会话的updated_at时间
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            是否更新成功
        """
        stmt = update(Conversation).where(Conversation.id == conversation_id).values(updated_at=datetime.now())
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def update_conversation_title(
        self,
        conversation_id: int,
        title: str
    ) -> bool:
        """
        更新会话标题
        
        Args:
            conversation_id: 会话ID
            title: 新标题
        
        Returns:
            是否更新成功
        """
        # TODO: 实现更新标题
        # 提示：参考update_conversation_time，只是values改成title
        stmt = update(Conversation).where(Conversation.id == conversation_id).values(title=title)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    # ========== messages表方法 ==========
    
    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        tokens: Optional[int] = None
    ) -> Message:
        """
        添加单条消息
        
        Args:
            conversation_id: 会话ID
            role: 角色（user/assistant/system）
            content: 消息内容
            tokens: Token数（可选）
        
        Returns:
            Message对象
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens=tokens
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message
    
    async def add_messages(
        self,
        conversation_id: int,
        messages_list: List[Dict]
    ) -> List[Message]:
        """
        批量添加消息
        
        Args:
            conversation_id: 会话ID
            messages_list: 消息列表
                [{"role": "user", "content": "...", "tokens": 10}, ...]
        
        Returns:
            Message对象列表
        """
        # 创建Message对象列表
        messages = [
            Message(
                conversation_id=conversation_id,
                role=msg["role"],
                content=msg["content"],
                tokens=msg.get("tokens")
            )
            for msg in messages_list
        ]
        
        # 批量添加
        self.session.add_all(messages)
        await self.session.commit()
        
        # 刷新所有对象（获取数据库生成的id和时间戳）
        for msg in messages:
            await self.session.refresh(msg)
        
        return messages
    
    async def query_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """
        查询消息
        
        Args:
            conversation_id: 会话ID
            limit: 限制数量（None=全部）
            offset: 偏移量（分页用）
        
        Returns:
            Message对象列表
        """
        stmt = select(Message).where(Message.conversation_id == conversation_id)
        stmt = stmt.order_by(Message.created_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        stmt = stmt.offset(offset)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_tokens(self, conversation_id: int) -> Dict[str, int]:
        """
        获取Token统计
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            {"total": 总Token数, "user": 用户Token数, "assistant": AI Token数}
        """
        from sqlalchemy import func
    
        stmt = select(
            Message.role,
            func.sum(Message.tokens).label('token_count')
        ).where(
            Message.conversation_id == conversation_id
        ).group_by(Message.role)
        
        result = await self.session.execute(stmt)
        rows = result.all()
        
        token_stats = {"total": 0, "user": 0, "assistant": 0, "system": 0}
        for role, token_count in rows:
            if token_count is not None:
                token_stats[role] = int(token_count)
                token_stats["total"] += int(token_count)
    
        return token_stats
    
    async def delete_messages(self, conversation_id: int) -> bool:
        """
        删除会话的所有消息
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            是否删除成功
        """
        stmt = delete(Message).where(Message.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    # ========== user_profiles表方法 ==========

    async def upsert_profile(
        self,
        conversation_id: int,
        profile_data: Dict[str, str]
    ) -> bool:
        """
        保存/更新用户画像（PostgreSQL UPSERT）
        """
        for key, value in profile_data.items():
            stmt = pg_insert(UserProfile).values(
                conversation_id=conversation_id,
                profile_key=key,
                profile_value=value
            ).on_conflict_do_update(
                index_elements=['conversation_id', 'profile_key'],
                set_={
                    'profile_value': value,
                    'updated_at': datetime.now()
                }
            )
            await self.session.execute(stmt)
        
        await self.session.commit()
        return True 
    
    async def get_profile(self, conversation_id: int) -> Dict[str, str]:
        """
        获取用户画像
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            画像字典，{"name": "Tom", "age": "28"}
        """
        stmt = select(UserProfile).where(UserProfile.conversation_id == conversation_id)
        result = await self.session.execute(stmt)
        profiles = result.scalars().all()
        
        profile_dict = {profile.profile_key: profile.profile_value for profile in profiles}
        return profile_dict
    
    # ========== summaries表方法 ==========
    
    async def save_summary(
        self,
        conversation_id: int,
        time_range: str,
        summary_text: str
    ) -> Summary:
        """
        保存摘要
        
        Args:
            conversation_id: 会话ID
            time_range: 时间范围（例如："2024-12-05_10:00"）
            summary_text: 摘要内容
        
        Returns:
            Summary对象
        """
        summary = Summary(
            conversation_id=conversation_id,
            time_range=time_range,
            summary_text=summary_text
        )
        self.session.add(summary)
        await self.session.commit()
        await self.session.refresh(summary)
        return summary
    
    async def get_summaries(self, conversation_id: int) -> List[Summary]:
        """
        获取所有摘要
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            Summary对象列表（按时间排序）
        """
        stmt = select(Summary).where(Summary.conversation_id == conversation_id).order_by(Summary.created_at)
        result = await self.session.execute(stmt)
        return result.scalars().all()