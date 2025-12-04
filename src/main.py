import os
from dotenv import load_dotenv
from llm.deepseek import DeepSeekLLM
from chatbot import MemoryChatbotWithCompressor
from memory.compressor import (
    SlidingWindowCompressor,
    LLMSummaryCompressor,
    HybridCompressor,
    TokenBasedCompressor
)


def print_header():
    """打印程序头部"""
    print("\n" + "=" * 70)
    print("🧠 Memory Chatbot v0.3 - 带压缩功能")
    print("=" * 70)
    print("功能说明:")
    print("  - 支持4种压缩策略")
    print("  - 自动管理对话上下文")
    print("  - 实时显示统计信息")
    print("\n命令:")
    print("  'stats'    - 查看统计信息")
    print("  'clear'    - 清空记忆")
    print("  'compress' - 查看压缩策略")
    print("  'switch'   - 切换压缩策略")
    print("  'quit'     - 退出程序")
    print("=" * 70 + "\n")


def print_stats(bot: MemoryChatbotWithCompressor) -> None:
    """打印统计信息"""
    stats = bot.get_stats()
    print(f"\n{'='*70}")
    print(f"📊 统计信息")
    print(f"{'='*70}")
    print(f"对话轮数: {stats['turns']}")
    print(f"消息数量: {stats['messages']}")
    print(f"总Token数: {stats['total_tokens']}")
    print(f"记忆状态: {'已满' if stats['is_full'] else '未满'}")
    print(f"\n压缩功能: {'启用' if stats['compression_enabled'] else '禁用'}")
    if stats['compression_enabled']:
        print(f"压缩策略: {stats['compressor']}")
        print(f"压缩次数: {stats['total_compressions']}")
        print(f"节省Token: {stats['tokens_saved']}")
        if stats['tokens_saved'] > 0:
            saved_cost = stats['tokens_saved'] / 1_000_000 * 0.14  # DeepSeek价格
            print(f"节省成本: ${saved_cost:.6f}")
    print(f"{'='*70}\n")


def choose_compressor(llm) -> tuple:
    """
    让用户选择压缩策略
    
    Returns:
        (compressor, enable_compression)
    """
    print("\n选择压缩策略:")
    print("  1. 不使用压缩（最快，但可能丢失历史）")
    print("  2. 滑动窗口（快速，固定保留最近N轮）")
    print("  3. LLM摘要（智能，保留语义信息）")
    print("  4. 混合策略（推荐，自动选择最优）")
    print("  5. Token动态（精确控制token数量）")
    
    while True:
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == "1":
            return None, False
        elif choice == "2":
            keep_turns = int(input("保留最近几轮？(默认5): ") or "5")
            return SlidingWindowCompressor(keep_turns=keep_turns), True
        elif choice == "3":
            keep_turns = int(input("保留最近几轮完整对话？(默认3): ") or "3")
            return LLMSummaryCompressor(llm, keep_recent_turns=keep_turns), True
        elif choice == "4":
            threshold = int(input("摘要触发阈值（轮数）？(默认10): ") or "10")
            keep_turns = int(input("保留最近几轮？(默认3): ") or "3")
            return HybridCompressor(llm, threshold_turns=threshold, keep_recent_turns=keep_turns), True
        elif choice == "5":
            max_tokens = int(input("最大token数？(默认4000): ") or "4000")
            return TokenBasedCompressor(llm.count_tokens, max_tokens=max_tokens), True
        else:
            print("无效选择，请重新输入")


def main():
    """命令行聊天程序"""
    # 加载环境变量
    load_dotenv()
    
    # 初始化LLM
    print("正在初始化...")
    llm = DeepSeekLLM()
    
    # 选择压缩策略
    compressor, enable_compression = choose_compressor(llm)
    
    # 初始化Chatbot
    bot = MemoryChatbotWithCompressor(
        llm,
        system_prompt="你是一个友好、乐于助人的AI助手。请记住用户告诉你的信息。",
        max_turns=20,
        compressor=compressor,
        enable_compression=enable_compression,
        compression_trigger=10
    )
    
    # 打印头部
    print_header()
    
    if enable_compression:
        print(f"✅ 已启用压缩: {compressor.__class__.__name__}\n")
    else:
        print(f"⚠️  未启用压缩\n")
    
    # 对话循环
    turn = 0
    while True:
        try:
            # 获取用户输入
            user_input = input(f"You [{turn}]: ").strip()
            
            # 特殊命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！👋")
                break
            
            if user_input.lower() == 'stats':
                print_stats(bot)
                continue
            
            if user_input.lower() == 'clear':
                bot.clear_history()
                turn = 0
                print("\n✅ 记忆已清空\n")
                continue
            
            if user_input.lower() == 'compress':
                if bot.enable_compression:
                    print(f"\n当前压缩策略: {bot.compressor.__class__.__name__}")
                    print(f"已执行压缩次数: {bot.stats['compressions']}")
                    print(f"已节省Token数: {bot.stats['tokens_saved']}\n")
                else:
                    print("\n未启用压缩\n")
                continue
            
            if user_input.lower() == 'switch':
                compressor, enable = choose_compressor(llm)
                if enable:
                    bot.set_compressor(compressor)
                    print(f"\n✅ 已切换到: {compressor.__class__.__name__}\n")
                else:
                    bot.enable_compression = False
                    print(f"\n✅ 已禁用压缩\n")
                continue
            
            # 空输入
            if not user_input:
                continue
            
            # 获取回复
            response = bot.chat(user_input)
            print(f"Bot [{turn}]: {response}\n")
            
            turn += 1
            
            # 每5轮显示一次统计
            if turn % 5 == 0:
                print_stats(bot)
            
        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()