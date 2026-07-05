# -*- coding: utf-8 -*-
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain import hub
from typing import List, Union

# --- DeepSeek API 配置 ---
# 请替换为你的 DeepSeek API 密钥
API_KEY = "xxxxx"
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# --- 步骤1: 初始化 ChatOpenAI ---
# 虽然我们用的是DeepSeek，但它兼容OpenAI的API格式，所以可以使用ChatOpenAI类
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.8,
    max_tokens=300
)

# --- 步骤2: 定义我们的工具 ---
@tool
def multiply(a: int, b: int) -> int:
    """用于计算两个整数的乘积。"""
    print(f"正在执行乘法: {a} * {b}")

    return a * b

@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。"""
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"

# 将工具列表放入一个变量
tools = [multiply, search_weather]

# 通用的执行函数，用于运行agent并打印结果
def run_agent_and_print(agent_executor, query):
    """一个通用函数，用于运行Agent并打印结果。"""
    print(f"--- 运行Agent，查询: {query} ---")
    response = agent_executor.invoke({"input": query})
    print(f"\n--- Agent响应: ---")
    print(response.get("output", "没有找到输出。"))
    print("-" * 30 + "\n")

# 步骤1: 自定义一个 Agent 的 Prompt 模板
# 这个 Prompt 告诉 LLM 它的角色，以及如何使用工具。
# 注意 {tools}, {tool_names}, {input} 是必需的占位符。
tool_use_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的AI助手，可以访问和使用各种工具来回答问题。请根据用户的问题，决定是否需要调用工具。当需要调用工具时，请使用正确的JSON格式。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}") # 这个占位符用于保存 Agent 的思考过程和工具调用历史
])

# 步骤2: 创建一个 LLM 能够识别和使用的 Agent
# 使用 create_tool_calling_agent 函数，它能让 LLM 自动判断何时以及如何调用工具
tool_calling_agent = create_tool_calling_agent(llm, tools, tool_use_prompt)

# 步骤3: 创建 Agent Executor
# AgentExecutor 负责 Agent 和工具之间的协调
tool_use_executor = AgentExecutor(
    agent=tool_calling_agent,
    tools=tools,
    verbose=True  # 开启 verbose 模式，可以打印详细的执行过程
)

# 步骤4: 运行并测试 Agent
if __name__ == "__main__":
    run_agent_and_print(tool_use_executor, "上海今天的天气怎么样？")
    run_agent_and_print(tool_use_executor, "30乘以5等于多少？ 上海天气怎么样")