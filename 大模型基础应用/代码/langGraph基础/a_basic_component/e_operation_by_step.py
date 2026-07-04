import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain.agents import create_agent

from deepseek_connection import deepseek_client


#第一步：初始化模型和工具
def getID():
    """explain your identity"""
    return f"我是凯瑞汽车的档案管理助手"

# model=ChatOpenAI(
#         model="deepseek-v4-flash",              # 模型名
#         api_key = str(os.environ["DEEPSEEK_API_KEY"]),
#         base_url="https://api.deepseek.com", # 换成任意兼容接口
#     )

model=ChatOllama(model="qwen2.5:7b")

llm = create_agent(
    model =model,
    tools=[getID],
    system_prompt="You are a helpful assistant"
)

def chatbot(state):
    return llm.invoke(state)

#第二步：定义图和状态
class State(TypedDict):
    messages: Annotated[list, add_messages]
graphBuilder = StateGraph(dict)

#第三步：定义节点
graphBuilder.add_node("chatbot", chatbot)

#第四步：定义边和出入口
graphBuilder.add_edge(START, "chatbot")
graphBuilder.add_edge("chatbot",END)

#第五步：编译图
graph = graphBuilder.compile()

#第六步：执行图
userInput = input("User: ")
state:State = {"messages": [{"role": "user", "content": userInput}]}
result = graph.invoke(state)
print("Assistant:"+str(result["messages"][-1].content))
