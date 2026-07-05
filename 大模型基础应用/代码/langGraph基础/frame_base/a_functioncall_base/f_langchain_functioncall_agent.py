from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import Config

# 初始化配置
conf = Config()

# 第一步：定义工具
@tool
def add(a: int, b: int) -> int:
    """将数字 a 与数字 b 相加"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """将数字 a 与数字 b 相乘"""
    return a * b

# 工具列表
tools = [add, multiply]
print(tools)

# 第二步：初始化模型
llm = ChatOpenAI(
    model=conf.model_name,
    api_key=conf.api_key,
    base_url="https://gateway.ai.cloudflare.com/v1/d2cbfe461e343906da9615cbceab35c6/ai2525/deepseek",
    temperature=0,  # 确保输出更可控
    streaming = False
)

# 第三步：定义 Agent 的 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是可以利用提供的工具进行数学计算的助手。请清晰简洁地回答。"),
    MessagesPlaceholder(variable_name="messages"),  # 用户输入和历史消息
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent 的中间思考步骤
])


# 第四步：创建 Agent
#create_tool_calling_agent 创建 Agent，自动处理工具调用逻辑。
agent = create_tool_calling_agent(llm, tools, prompt)

# 第五步：创建 AgentExecutor
#AgentExecutor 负责执行 Agent 的完整工作流，包括工具调用、结果处理和错误管理。
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印中间步骤，便于调试
    handle_parsing_errors=True  # 自动处理解析错误
)

# 第六步：执行查询
query = "2+3等于多少？ 11*2是多少"
response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})

# 第七步：打印最终结果
print("最终结果：")
print(response["output"])



"""
被 @tool 装饰的函数会自动生成工具的元数据，
包括函数名、描述和参数信息。

功能上与tools定义的json是等价的
"""