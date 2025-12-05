"""
数据库初始化脚本

用法：
    python scripts/init_db.py
"""
# 在 scripts/init_db.py 里修改
import os

# 获取当前用户名
username = os.getenv("USER")  

DATABASE_URL = f"postgresql+asyncpg://{username}@localhost:5432/memory_chatbot"
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.database import DatabaseManager


async def init_database():
    """初始化数据库"""
    
    print("=" * 60)
    print("初始化数据库")
    print("=" * 60)
    print(f"数据库URL: {DATABASE_URL}")
    print()
    
    # 创建数据库管理器
    db_manager = DatabaseManager(DATABASE_URL, echo=True)
    
    try:
        print("正在创建表...")
        await db_manager.create_tables()
        print("✅ 表创建成功！")
        print()
        
        print("创建的表：")
        print("  - conversations")
        print("  - messages")
        print("  - user_profiles")
        print("  - summaries")
        print()
        
    except Exception as e:
        print(f"❌ 创建表失败: {e}")
        raise
    
    finally:
        await db_manager.close()
        print("数据库连接已关闭")


if __name__ == "__main__":
    asyncio.run(init_database())