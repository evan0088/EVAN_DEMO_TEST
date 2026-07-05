from python_a2a import Task, TaskStatus, TaskState, Message, MessageRole, TextContent

# 创建任务
message = Message(content=TextContent(text="查询天气"), role=MessageRole.USER)
task = Task(message=message.to_dict())
print(task)

# 处理中更新状态
task.status = TaskStatus(state=TaskState.WAITING, message={"info": "调用工具"})

# 完成任务
# task.artifacts = [{"parts": [{"type": "text", "text": "晴天"}]}]
task.status = TaskStatus(state=TaskState.COMPLETED)

# 序列化输出
print(task.to_dict())