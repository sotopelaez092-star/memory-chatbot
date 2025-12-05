from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, Integer, BigInteger, ForeignKey, TIMESTAMP, UniqueConstraint, func, CheckConstraint
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """所有模型的基类"""
    pass


class Conversation(Base):
    """会话表模型"""
    __tablename__ = "conversations"
    
    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # 用户ID和会话ID
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()  # 数据库层面的默认值
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()  # 更新时自动更新
    )
    
    # 可选标题
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # 联合唯一约束
    __table_args__ = (
        UniqueConstraint('user_id', 'session_id', name='uq_user_session'),
    )
    
    # 关系（一个会话有多条消息）
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan"  # 删除会话时自动删除消息
    )
    
    # 关系（一个会话有多条画像）
    profiles: Mapped[List["UserProfile"]] = relationship(
        "UserProfile",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    # 关系（一个会话有多条摘要）
    summaries: Mapped[List["Summary"]] = relationship(
        "Summary",
        back_populates="conversation",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, session_id={self.session_id})>"

class Message(Base):
    """消息表模型"""
    __tablename__ = "messages"
    
    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # 外键（关联到conversations表）
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 角色（user/assistant/system）
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False
    )
    
    # 消息内容
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # Token数（可选）
    tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        index=True  # 常用于按时间查询
    )
    
    # 反向关系（关联到conversation）
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages"
    )
    
    # CHECK约束
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name='check_role'
        ),
    )
    
    def __repr__(self):
        return f"<Message(id={self.id}, role={self.role}, content={self.content[:30]}...)>"


class UserProfile(Base):
    """用户画像表模型"""
    __tablename__ = "user_profiles"
    
    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # 外键（关联到conversations表）
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 画像键（例如：name, age, location）
    profile_key: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    
    # 画像值
    profile_value: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # 更新时间
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # 反向关系
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="profiles"
    )
    
    # 联合唯一约束（同一个会话的同一个key只能有一条记录）
    __table_args__ = (
        UniqueConstraint('conversation_id', 'profile_key', name='uq_conversation_profile'),
    )
    
    def __repr__(self):
        return f"<UserProfile(key={self.profile_key}, value={self.profile_value})>"


class Summary(Base):
    """摘要表模型"""
    __tablename__ = "summaries"
    
    # 主键
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    # 外键（关联到conversations表）
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 时间范围标识（例如：2024-12-05_10:00）
    time_range: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )
    
    # 摘要内容
    summary_text: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )
    
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    
    # 反向关系
    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="summaries"
    )
    
    # 联合唯一约束（同一个会话的同一个时间范围只能有一条摘要）
    __table_args__ = (
        UniqueConstraint('conversation_id', 'time_range', name='uq_conversation_timerange'),
    )
    
    def __repr__(self):
        return f"<Summary(time_range={self.time_range}, text={self.summary_text[:50]}...)>"