from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import os
from langsmith import traceable

os.environ['LANGSMITH_TRACING']='true'
os.environ['LANGSMITH_ENDPOINT']='https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY']='xxxxx'
os.environ['LANGSMITH_PROJECT']='test1'
os.environ['OPENAI_API_KEY']='<your-openai-api-key>'

agent = create_agent(
    model=ChatOllama(model="qwen2.5:7b"),
    tools=[],
    prompt=""
)

@traceable
def callAgent():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "天空为什么是蓝色的？"}]}
    )
    print(result["messages"][-1].content)


callAgent()