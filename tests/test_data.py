"""
测试数据集
包含不同长度的真实对话场景
"""

def get_short_conversation():
    """短对话：5轮"""
    return [
        {"role": "user", "content": "我叫Tom"},
        {"role": "assistant", "content": "你好Tom！很高兴认识你。"},
        {"role": "user", "content": "我在上海工作"},
        {"role": "assistant", "content": "上海是个很国际化的城市！你在哪个行业工作？"},
        {"role": "user", "content": "我是AI工程师"},
        {"role": "assistant", "content": "AI工程师是个很有前景的职业！"},
        {"role": "user", "content": "我研究Agent"},
        {"role": "assistant", "content": "Agent是很前沿的方向，有什么具体研究内容吗？"},
        {"role": "user", "content": "多Agent协作"},
        {"role": "assistant", "content": "多Agent协作确实很有挑战性！"},
    ]


def get_medium_conversation():
    """中等对话：10轮"""
    messages = get_short_conversation()
    messages.extend([
        {"role": "user", "content": "我遇到了Agent通信的问题"},
        {"role": "assistant", "content": "通信问题很常见，具体是什么问题？"},
        {"role": "user", "content": "消息延迟太高"},
        {"role": "assistant", "content": "消息延迟可以考虑优化消息队列。"},
        {"role": "user", "content": "用什么消息队列比较好？"},
        {"role": "assistant", "content": "可以考虑RabbitMQ或Redis，看你的场景。"},
        {"role": "user", "content": "Redis性能怎么样？"},
        {"role": "assistant", "content": "Redis非常快，单机QPS可以达到10万+。"},
        {"role": "user", "content": "有什么坑需要注意？"},
        {"role": "assistant", "content": "需要注意内存管理和持久化配置。"},
    ])
    return messages


def get_long_conversation():
    """长对话：20轮"""
    messages = get_medium_conversation()
    messages.extend([
        {"role": "user", "content": "持久化有哪些方案？"},
        {"role": "assistant", "content": "主要有RDB和AOF两种方案。"},
        {"role": "user", "content": "RDB和AOF有什么区别？"},
        {"role": "assistant", "content": "RDB是快照，AOF是日志，各有优劣。"},
        {"role": "user", "content": "我应该用哪个？"},
        {"role": "assistant", "content": "看你的场景，对数据安全性要求高就用AOF。"},
        {"role": "user", "content": "AOF会影响性能吗？"},
        {"role": "assistant", "content": "会有一定影响，但可以通过配置优化。"},
        {"role": "user", "content": "具体怎么优化？"},
        {"role": "assistant", "content": "可以调整fsync策略，比如everysec。"},
        {"role": "user", "content": "everysec是什么意思？"},
        {"role": "assistant", "content": "每秒钟同步一次到磁盘，平衡了性能和安全。"},
        {"role": "user", "content": "明白了，还有其他建议吗？"},
        {"role": "assistant", "content": "可以考虑主从复制和哨兵模式提高可用性。"},
        {"role": "user", "content": "主从复制怎么配置？"},
        {"role": "assistant", "content": "在从节点配置文件中添加slaveof主节点IP和端口。"},
        {"role": "user", "content": "哨兵模式是干什么的？"},
        {"role": "assistant", "content": "哨兵模式用于监控主节点，自动故障转移。"},
        {"role": "user", "content": "听起来很复杂"},
        {"role": "assistant", "content": "确实需要一些学习成本，但值得投入。"},
        {"role": "user", "content": "好的，谢谢你的建议"},
        {"role": "assistant", "content": "不客气！有问题随时问我。"},
    ])
    return messages


def get_very_long_conversation():
    """超长对话：30轮"""
    messages = get_long_conversation()
    messages.extend([
        {"role": "user", "content": "回到Agent的话题"},
        {"role": "assistant", "content": "好的，我们继续聊Agent。"},
        {"role": "user", "content": "Agent之间如何协调？"},
        {"role": "assistant", "content": "可以使用中心化的协调器或分布式共识算法。"},
        {"role": "user", "content": "分布式共识算法有哪些？"},
        {"role": "assistant", "content": "常见的有Paxos、Raft、ZAB等。"},
        {"role": "user", "content": "Raft算法好理解吗？"},
        {"role": "assistant", "content": "Raft算法相对Paxos更容易理解和实现。"},
        {"role": "user", "content": "能简单介绍下Raft吗？"},
        {"role": "assistant", "content": "Raft包含Leader选举、日志复制、安全性保证三个部分。"},
        {"role": "user", "content": "Leader选举是怎么实现的？"},
        {"role": "assistant", "content": "通过心跳和选举超时机制，节点投票选出Leader。"},
        {"role": "user", "content": "如果Leader挂了怎么办？"},
        {"role": "assistant", "content": "其他节点检测到超时后会发起新的选举。"},
        {"role": "user", "content": "这个过程需要多久？"},
        {"role": "assistant", "content": "通常几百毫秒到几秒，取决于配置。"},
        {"role": "user", "content": "这样不会影响服务吗？"},
        {"role": "assistant", "content": "会有短暂的不可用，但这是分布式系统的权衡。"},
        {"role": "user", "content": "有办法减少影响吗？"},
        {"role": "assistant", "content": "可以优化超时时间和使用更多的副本。"},
        {"role": "user", "content": "我的Agent系统需要用Raft吗？"},
        {"role": "assistant", "content": "要看你的系统规模和一致性要求。"},
        {"role": "user", "content": "如果就3-5个Agent呢？"},
        {"role": "assistant", "content": "3-5个Agent可能不需要Raft，简单的主从就够了。"},
        {"role": "user", "content": "好的，我先用简单方案"},
        {"role": "assistant", "content": "这是个明智的选择，先从简单开始！"},
    ])
    return messages