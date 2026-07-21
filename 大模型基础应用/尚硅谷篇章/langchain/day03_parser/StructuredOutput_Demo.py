# 课程外

"""
with_structured_output() 演示
功能: 直接通过模型API的JSON模式约束，让模型输出结构化的数据

两种方式:
1. Pydantic BaseModel —— 带类型校验、字段描述、校验器
2. TypedDict —— 轻量，无需额外依赖

对比 JsonOutputParserDemo.py:
  旧方式: JsonOutputParser() → 靠提示词告诉模型"输出JSON"，稳定性依赖模型
  新方式: with_structured_output() → API 层面约束输出必须是合法 JSON，更稳定
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from loguru import logger
import os

load_dotenv(encoding='utf-8')

# ============================================================
# 1. 定义数据结构
# ============================================================

# --- 方式A: Pydantic BaseModel（带校验）---
class News(BaseModel):
    """新闻结构化模型"""
    title: str = Field(description="新闻标题")
    time: str = Field(description="新闻发生的时间")
    person: str = Field(description="新闻涉及的人物")
    event: str = Field(description="发生的具体事件")

# --- 方式B: TypedDict（轻量）---
class Weather(TypedDict):
    city: Annotated[str, "城市名称"]
    temperature: Annotated[str, "温度，如25°C"]
    condition: Annotated[str, "天气状况，如晴、多云、雨"]


# ============================================================
# 2. 初始化模型
# ============================================================
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ============================================================
# 3. 绑定结构化输出（with_structured_output）
# ============================================================

# 方式A: 绑定 Pydantic 模型
news_model = model.with_structured_output(News)

# 方式B: 绑定 TypedDict
weather_model = model.with_structured_output(Weather)

# ============================================================
# 4. 调用测试
# ============================================================

print("=" * 60)
print("【案例1】with_structured_output + Pydantic BaseModel")
print("=" * 60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个新闻助手，请根据用户要求生成新闻"),
    ("human", "请生成一条关于{topic}的新闻")
])

messages = prompt.format_messages(topic="小米SU7跑车发布")
result1 = news_model.invoke(messages)

logger.info(f"原始消息:\n{messages}")
logger.info(f"输出类型: {type(result1)}")
logger.info(f"结构化结果: {result1}")
logger.info(f"title: {result1.title}")
logger.info(f"time: {result1.time}")
logger.info(f"person: {result1.person}")
logger.info(f"event: {result1.event}")

print("\n")
print("=" * 60)
print("【案例2】with_structured_output + TypedDict")
print("=" * 60)

messages2 = [{"role": "user", "content": "查询北京、上海、深圳三个城市的天气"}]
result2 = weather_model.invoke(messages2)

logger.info(f"输出类型: {type(result2)}")
logger.info(f"结构化结果: {result2}")
if isinstance(result2, dict):
    logger.info(f"city: {result2.get('city')}")
    logger.info(f"temperature: {result2.get('temperature')}")
    logger.info(f"condition: {result2.get('condition')}")

print("\n")
print("=" * 60)
print("【对比】旧方式 vs 新方式")
print("=" * 60)
print("旧方式 JsonOutputParser():")
print("  → 在提示词里写'结果返回json格式，q字段...a字段...'")
print("  → 模型可能不遵守，输出格式不合法时解析报错")
print("新方式 with_structured_output():")
print("  → 不需要在提示词里手写格式说明")
print("  → API 层面强制 JSON 输出，模型无法输出非法格式")
print("  → 直接返回解析好的 Python 对象，无需额外 parser")
