"""
模型参数演示
"""
import os
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=1.0
)

# 3.调用模型
for x in range(3):
    print(model.invoke("写一句关于春天的词,14字以内").content)