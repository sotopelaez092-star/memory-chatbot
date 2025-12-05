"""
Lua脚本集合

用于Redis的原子操作，解决并发问题
"""

# 添加消息到Sorted Set（原子操作）
ADD_MESSAGE_SCRIPT = """
local messages_key = KEYS[1]
local message = ARGV[1]
local timestamp = tonumber(ARGV[2])
local max_count = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- 检查是否有相同timestamp（并发情况）
-- 如果有，给timestamp加0.001避免覆盖
while redis.call('ZSCORE', messages_key, message) do
    timestamp = timestamp + 0.001
end

-- 添加消息
redis.call('ZADD', messages_key, timestamp, message)

-- 限制数量（只保留最近N条）
local count = redis.call('ZCARD', messages_key)
if count > max_count then
    redis.call('ZREMRANGEBYRANK', messages_key, 0, count - max_count - 1)
end

-- 设置过期时间
redis.call('EXPIRE', messages_key, ttl)

-- 返回当前消息总数
return count
"""

# 获取消息 + 更新访问时间
GET_MESSAGES_SCRIPT = """
local messages_key = KEYS[1]
local meta_key = KEYS[2]
local limit = tonumber(ARGV[1])
local current_time = ARGV[2]

-- 获取最近N条消息
local messages = redis.call('ZRANGE', messages_key, -limit, -1)

-- 更新最后访问时间
redis.call('HSET', meta_key, 'last_access', current_time)

-- 返回消息列表
return messages
"""

# 更新用户画像（增量更新）
UPDATE_PROFILE_SCRIPT = """
local profile_key = KEYS[1]
local ttl = tonumber(ARGV[1])

-- 从ARGV[2]开始，每两个参数是一对 field-value
local field_count = (#ARGV - 1) / 2
for i = 0, field_count - 1 do
    local field = ARGV[2 + i * 2]
    local value = ARGV[3 + i * 2]
    redis.call('HSET', profile_key, field, value)
end

-- 设置过期时间
redis.call('EXPIRE', profile_key, ttl)

return 1
"""