#pip install langchain-community
#pip install dashscope

import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_qwq import ChatQwen

load_dotenv(encoding='utf-8')


chatLLM = ChatTongyi(
    model="qwen-plus",   # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    streaming=True
)


llm = ChatQwen(
    model="qwen-flash",
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    streaming=True,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
res=llm.stream(input="你是谁")

for r in res:
    print(r.content,end="")

# 打印结果
print(chatLLM.invoke("你是谁"))

print("*" * 60)

res = chatLLM.stream([HumanMessage(content="你好，你是谁")], streaming=True)
for r in res:
    print(r.content,end="")
