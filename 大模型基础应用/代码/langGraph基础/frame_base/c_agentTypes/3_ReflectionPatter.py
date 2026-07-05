# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- DeepSeek API 配置 ---
# 请替换为你的 DeepSeek API 密钥
API_KEY = "xxxx"
API_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

# --- 步骤1: 初始化 ChatOpenAI ---
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.8,
    max_tokens=300
)

# --- 步骤2: 自定义 Prompt ---

# 2.1 初始响应 Prompt: 用于生成第一次的回答
initial_response_prompt = ChatPromptTemplate.from_template(
    "请根据以下问题给出你的初步回答: {question}"
)
initial_response_chain = initial_response_prompt | llm | StrOutputParser()

# 2.2 反思 Prompt: 用于接收用户反馈并优化回答
reflection_prompt = ChatPromptTemplate.from_template(
    """你是一个专业的、善于反思的AI助手。你之前给出了以下回答：
---
{previous_response}
---
现在，你收到了用户对你的回答给出的反馈：
---
{user_feedback}
---
请根据用户的反馈，认真反思你之前的回答，并生成一个更准确、更完善的新回答。
新回答:"""
)
reflection_chain = reflection_prompt | llm | StrOutputParser()


# --- 步骤3: 模拟反射过程 ---
def reflect_and_refine(query: str, feedback: str):
    """模拟一个完整的反射过程，从初始响应到优化后的响应。"""

    print("--- 启动反射模式 ---")
    print(f"用户查询: {query}")

    # LLM 生成初步响应
    print("\n生成初步响应...")
    initial_response = initial_response_chain.invoke({"question": query})
    print(f"LLM 初步响应:\n{initial_response}")

    # 模拟用户反馈
    print(f"\n用户反馈:\n{feedback}")

    # LLM 进行反思，并生成新的回答
    print("\nLLM 正在反思并生成新响应...")
    refined_response = reflection_chain.invoke({
        "previous_response": initial_response,
        "user_feedback": feedback
    })

    print("\n--- LLM 经过反思后的新响应 ---")
    print(refined_response)

    return refined_response


# --- 步骤4: 运行并测试 ---
if __name__ == "__main__":
    # 模拟用户查询
    initial_question = "请用一句话介绍一下 LangChain。"
    # 模拟用户反馈，指出初步回答的不足
    user_feedback_text = "你的回答太简单了，请更详细地解释一下 LangChain 的核心概念，比如 Agent 和 Chain 的区别。"

    # 运行反射过程
    reflect_and_refine(initial_question, user_feedback_text)