# LangChain0.3版本使用方式,了解即可，目前再用
from langchain_anthropic import ChatAnthropic
# 1.导入依赖
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
# follow  official https://docs.langchain.com/oss/python/integrations/providers/overview#popular-providers

# 第1版：硬编码写死
# llm = ChatOpenAI(
#     model="qwen-plus",
#     # 硬编码写死
#     api_key="你自己的api-key",  # 平台提供的 API-Key
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )


# 第2版：配置进环境变量
# llm = ChatOpenAI(
#     model="qwen-plus",
#     # 配置进环境变量
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# 第3版：通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
# 加载.env文件中的环境变量（指定编码，避免中文乱码）
load_dotenv(encoding='utf-8')
#
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     # 配置进环境变量
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

llm = ChatAnthropic(
   model_name="deepseek-chat"
)

# llm = ChatDeepSeek(
#     model="deepseek-chat",
# )

# # 2.提供问题，并调用llm
response = llm.invoke("你是谁")

print(response)#元数据
print()
print(response.content)

print()

