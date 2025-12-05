"""
数据库引擎和会话管理
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine
)
from .models import Base


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_url: str, echo: bool = False):
        """
        初始化数据库管理器
        
        Args:
            database_url: 数据库连接URL
                格式：postgresql+asyncpg://user:password@host:port/database
            echo: 是否打印SQL语句（调试用）
        """
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=echo,
            pool_size=10,          # 连接池大小
            max_overflow=20,       # 超过pool_size的额外连接数
            pool_pre_ping=True,    # 每次使用前测试连接
        )
        
        # 会话工厂
        self.async_session_maker = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False  # 提交后对象不过期
        )
    
    async def create_tables(self):
        """创建所有表"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """删除所有表（慎用！）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    async def get_session(self) -> AsyncSession:
        """获取数据库会话"""
        async with self.async_session_maker() as session:
            yield session
    
    async def close(self):
        """关闭数据库引擎"""
        await self.engine.dispose()


# 全局数据库管理器实例（可选）
_db_manager: DatabaseManager | None = None


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """
    初始化全局数据库管理器
    
    Args:
        database_url: 数据库连接URL
        echo: 是否打印SQL
    
    Returns:
        DatabaseManager实例
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def get_database() -> DatabaseManager:
    """获取全局数据库管理器"""
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_manager