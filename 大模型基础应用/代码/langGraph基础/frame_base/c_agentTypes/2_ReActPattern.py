# -*- coding: utf-8 -*-
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

# --- DeepSeek API 配置 ---
API_KEY = "xxxx"
API_URL = "https://gateway.ai.cloudflare.com/v1/d2cbfe461e343906da9615cbceab35c6/ai2525/deepseek"
MODEL = "deepseek-chat"

# --- 步骤1: 初始化 ChatOpenAI ---
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_URL,
    temperature=0.8,
    max_tokens=300
)

# --- 步骤2: 定义工具 ---
@tool
def subtract(numbers_str: str) -> int:
    """
    用于计算两个整数的差。

    参数:
        numbers_str (str): 包含两个整数的字符串，用逗号分隔，例如："100,25"。

    返回:
        int: 两个整数的差。
    """
    print(f"正在执行减法: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(',')
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a - b
    except ValueError:
        return "输入的格式不正确，请确保是两个用逗号分隔的整数，例如：'100,25'"

@tool
def multiply(numbers_str: str) -> int:
    """用于计算两个整数的乘积。

    参数:
        numbers_str (str): 包含两个整数的字符串，用逗号分隔，例如："100,25"。

    返回:
        int: 两个整数的乘积。
    """
    print(f"正在执行乘法: {numbers_str}")
    try:
        a_str, b_str = numbers_str.split(',')
        a = int(a_str.strip())
        b = int(b_str.strip())
        return a * b
    except ValueError:
        return "输入的格式不正确，请确保是两个用逗号分隔的整数，例如：'100,25'"



@tool
def search_weather(city: str) -> str:
    """用于查询指定城市的实时天气。

    参数:
        city (str): 要查询天气的城市名称。

    返回:
        str: 该城市的天气信息。
    """
    print(f"正在查询天气: {city}")
    if "北京" in city:
        return "北京今天是晴天，气温25摄氏度。"
    elif "上海" in city:
        return "上海今天是阴天，有小雨，气温22摄氏度。"
    else:
        return f"抱歉，我没有'{city}'的天气信息。"

tools = [subtract,multiply, search_weather]

# --- 步骤3: 自定义 ReAct 风格的 Prompt ---
react_prompt_template = """你是一个有用的 AI 助手，可以访问以下工具：

{tools}

请根据用户输入一步步推理，并按以下规则操作：
1. 每次输出只能包含一个动作（Action 和 Action Input）或一个最终答案（Final Answer）。
2. 如果用户输入包含多个任务，依次处理每个任务，不要一次性输出所有步骤。
3. 每次行动前，说明你的思考（Thought），并选择合适的工具或直接给出最终答案。
4. 如果需要使用工具，格式必须为：
   Thought: [你的思考]
   Action: [工具名称]
   Action Input: [工具的输入参数，例如对于multiply工具，使用'100,25'格式]
5. 如果可以直接回答或所有任务都完成，格式为：
   Thought: [你的思考]
   Final Answer: [最终答案]

可用的工具名称有: {tool_names}

用户输入: {input}
{agent_scratchpad}
"""

react_prompt = ChatPromptTemplate.from_template(react_prompt_template)

# --- 步骤4: 创建 ReAct 风格的 Agent ---
react_agent = create_react_agent(llm, tools, react_prompt)

# --- 步骤5: 创建 Agent Executor ---
react_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 启用错误处理，自动重试解析错误
)

# --- 步骤6: 运行并测试 Agent ---
if __name__ == "__main__":
    # 测试用例1: 查询天气
    '''print("--- 运行Agent，询问: 该穿什么衣服出门？ ---")
    response_weather = react_executor.invoke({"input": "我在上海，今天出门该穿什么衣服？"})
    print(f"\n--- Agent响应: ---")
    print(response_weather.get("output", "没有找到输出。"))
    print("-" * 30 + "\n")'''

    # 测试用例2: 数学计算
    print("--- 运行Agent，查询: 整列火车还能装载多少名乘客？ ---")
    response_math = react_executor.invoke({"input": "整列火车总共25节，其中有一节是车头。每节车厢能够乘坐108名乘客，现在火车上已有1127名乘客，最多还能装载多少名乘客？"})
    print(f"\n--- Agent响应: ---")
    print(response_math.get("output", "没有找到输出。"))
    print("-" * 30 + "\n")
