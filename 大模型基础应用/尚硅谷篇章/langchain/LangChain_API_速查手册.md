# LangChain 三天课程 — API 速查手册

> 涵盖 Day01（入门）→ Day02（模型 I/O + Prompt）→ Day03（输出解析器）全部 API

---

## 目录

- [一、环境与初始化](#一环境与初始化)
- [二、模型调用（Model I/O）](#二模型调用model-io)
- [三、调用方式（invoke / stream / batch）](#三调用方式invoke--stream--batch)
- [四、消息类型（Messages）](#四消息类型messages)
- [五、PromptTemplate（提示词模板）](#五prompttemplate提示词模板)
- [六、ChatPromptTemplate（聊天提示词模板）](#六chatprompttemplate聊天提示词模板)
- [七、外部加载 Prompt](#七外部加载-prompt)
- [八、输出解析器（Output Parser）](#八输出解析器output-parser)
- [九、结构化输出（with_structured_output）](#九结构化输出with_structured_output)
- [十、类型注解（Annotated + Pydantic / TypedDict）](#十类型注解annotated--pydantic--typeddict)
- [附录：完整导入速查](#附录完整导入速查)

---

## 一、环境与初始化

### 1.1 dotenv 加载环境变量
> 📂 Demo：[GetEnvInfo.py](./day01_hello_word/GetEnvInfo.py) | [StandardDesc.py](./day01_hello_word/StandardDesc.py)

```python
from dotenv import load_dotenv
import os

load_dotenv(encoding='utf-8')          # 从 .env 文件加载

api_key = os.getenv("DEEPSEEK_API_KEY") # 读取指定变量
```

| API | 说明 |
|-----|------|
| `load_dotenv(encoding='utf-8')` | 加载 `.env` 文件中的环境变量到 `os.environ`，`encoding` 参数避免中文乱码 |
| `os.getenv("KEY")` | 读取环境变量，不存在则返回 `None` |


### 1.2 查看 LangChain 版本
> 📂 Demo：[LangChainV1.0.py](./day01_hello_word/LangChainV1.0.py) | [LangChainV0.3.py](./day01_hello_word/LangChainV0.3.py) | [LangChain_MoreV1.0.py](./day01_hello_word/LangChain_MoreV1.0.py)

```python
import langchain
import langchain_community

print(langchain.__version__)           # LangChain 核心版本
print(langchain_community.__version__)  # 社区包版本
print(langchain.__file__)              # 安装路径
```


---

## 二、模型调用（Model I/O） 

### 2.1 init_chat_model — 统一入口（⭐ 推荐，v1.0+） [📖 官方概述](https://docs.langchain.com/oss/python/langchain/models)
> 📂 Demo：[ModelIO_Init_chat_model.py](./day02_models_io/part1_模型调用入门/ModelIO_Init_chat_model.py)
[📖 参数概述](https://docs.langchain.com/oss/python/langchain/models#parameters)
```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="deepseek-chat",              # 模型名称
    model_provider="deepseek",          # 提供商，可省略（自动推导）
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,                    # 可选：温度参数
    max_tokens=2048                     # 可选：最大输出 Token
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str` | **必填**。模型名称，如 `"deepseek-chat"`、`"qwen-plus"`、`"gpt-4o"` |
| `model_provider` | `str` | 提供商标识：`"openai"`、`"deepseek"`、`"anthropic"` 等。匹配 base_url 时可省略 |
| `api_key` | `str` | API 密钥，建议从环境变量读取 |
| `base_url` | `str` | API 端点地址 |
| `temperature` | `float` | 控制输出随机性，0~2，值越高越随机 |
| `max_tokens` | `int` | 限制单次输出最大 Token 数 |

**智能推导机制**：当 `model="deepseek-chat"` + `base_url="https://api.deepseek.com"` 时，`model_provider` 可省略，函数内部自动匹配。


### 2.2 ChatOpenAI — 兼容 OpenAI 协议的模型（v0.3 风格）  [📖 官方概述](https://docs.langchain.com/oss/python/integrations/chat)
> 📂 Demo：[ModelIO_ChatOpenAI.py](./day02_models_io/part1_模型调用入门/ModelIO_ChatOpenAI.py) | [ModelIO_Params.py](./day02_models_io/part1_模型调用入门/ModelIO_Params.py)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,
    max_tokens=2048
)
```

> **用途**：所有兼容 OpenAI API 协议的厂商（DeepSeek、通义千问、SiliconFlow 等）均可通过 ChatOpenAI 调用。


### 2.3 ChatDeepSeek — DeepSeek 专用客户端 [📖 官方概述](https://docs.langchain.com/oss/python/integrations/chat)
> 📂 Demo：[ModelIO_DeepSeek.py](./day02_models_io/part1_模型调用入门/ModelIO_DeepSeek.py)

```python
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("deepseek-api"),
)
```

| 参数 | 说明 |
|------|------|
| `model` | 模型名，`"deepseek-chat"` 对应 V3（非思考模式），`"deepseek-reasoner"` 对应 R1 |
| `temperature` | 温度参数，DeepSeek 建议设为 0 |
| `max_tokens` | 最大输出，`None` 表示不限制 |
| `timeout` | 请求超时时间（秒），`None` 表示不限制 |
| `max_retries` | 失败重试次数 |

> **注意**：ChatDeepSeek 内置了默认 `base_url`，无需手动指定。


### 2.4 ChatOllama — 本地模型调用 [📖 官方概述](https://docs.langchain.com/oss/python/integrations/chat)
> 📂 Demo：[ModelIO_Ollama.py](./day02_models_io/part1_模型调用入门/ModelIO_Ollama.py) | [LangChain_Ollama.py](./day02_models_io/part2_Ollama_本地模型部署/LangChain_Ollama.py)

```python
from langchain_ollama import ChatOllama 

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3.5:0.8b",
    temperature=0,
)
```

| 参数 | 说明 |
|------|------|
| `base_url` | Ollama 服务地址，本地默认 `http://localhost:11434` |
| `model` | Ollama 中的模型名，如 `"qwen3.5:0.8b"`、`"llama3:8b"` |
| `temperature` | 温度参数 |

> **前提**：需先安装 Ollama 并 `ollama pull <模型名>`。


### 2.5 ChatTongyi — 通义千问原生客户端 [📖 阿里云概述](https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654)
> 📂 Demo：[ModelIO_Qwen.py](./day02_models_io/part1_模型调用入门/ModelIO_Qwen.py)

```python
from langchain_community.chat_models.tongyi import ChatTongyi

chatLLM = ChatTongyi(
    model="qwen-plus",
    streaming=True
)
```


### 2.6 ChatQwen — 通义千问新版客户端   [📖 官方概述](https://docs.langchain.com/oss/python/integrations/chat/qwen)
> 📂 Demo：[ModelIO_Qwen.py](./day02_models_io/part1_模型调用入门/ModelIO_Qwen.py)

```python
from langchain_qwq import ChatQwen

llm = ChatQwen(
    model="qwen-flash",
    max_tokens=3_000,
    timeout=None,
    max_retries=2,
    streaming=True,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
```

### 2.7 原生 OpenAI SDK 调用（对比参考）
> 📂 Demo：[ModelIO_OpenAI.py](./day02_models_io/part1_模型调用入门/ModelIO_OpenAI.py)

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("deepseek-api"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello,你是谁"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```


---

## 三、调用方式（invoke / stream / batch）

LangChain 模型对象是 **Runnable**，提供统一的调用接口。每种方法都有 **同步** / **异步** 两个版本。

### 3.1 概览表

| 方法 | 用途 | 返回类型 | 适用场景 |
|------|------|----------|----------|
| `invoke(input)` | 单次调用 | `AIMessage` | 一问一答 |
| `ainvoke(input)` | 异步单次调用 | `AIMessage` | FastAPI / 异步 Web 服务 |
| `batch(inputs)` | 同步批量调用 | `list[AIMessage]` | 批量处理少量问题 |
| `abatch(inputs)` | 异步批量调用 | `list[AIMessage]` | 大规模异步批量处理 |
| `stream(input)` | 同步流式调用 | `Iterator[AIMessageChunk]` | 实时打字效果 |
| `astream(input)` | 异步流式调用 | `AsyncIterator[AIMessageChunk]` | WebSocket / SSE 推送 |

### 3.2 invoke — 同步单次调用
> 📂 Demo：[LLM_Invoke.py](./day02_models_io/part03_prompt/invoke/LLM_Invoke.py)

```python
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 简单字符串
response = model.invoke("你是谁")
print(response.content)           # 纯文本回复
print(response.content_blocks)    # 结构化内容块

# 消息列表（带角色）
messages = [
    SystemMessage(content="你是一个法律助手，只回答法律问题"),
    HumanMessage(content="简单介绍下广告法")
]
response = model.invoke(messages)
```


### 3.3 ainvoke — 异步单次调用
> 📂 Demo：[LLM_aInvoke.py](./day02_models_io/part03_prompt/invoke/LLM_aInvoke.py)

```python
import asyncio

async def main():
    response = await model.ainvoke("解释一下LangChain是什么，简洁回答100字以内")
    print(response.content)

asyncio.run(main())
```

> **核心作用**：不阻塞主线程，适合大批量请求或 Web 服务（如 FastAPI）。


### 3.4 batch — 同步批量调用
> 📂 Demo：[LLM_Batch.py](./day02_models_io/part03_prompt/invoke/LLM_Batch.py)

```python
questions = [
    "什么是redis?简洁回答，字数控制在100以内",
    "Python的生成器是做什么的？简洁回答，字数控制在100以内",
    "解释一下Docker和Kubernetes的关系?简洁回答，字数控制在100以内"
]

response = model.batch(questions)   # 返回 list[AIMessage]

for q, r in zip(questions, response):
    print(f"问题：{q}\n回答：{r.content}\n")
```


### 3.5 abatch — 异步批量调用
> 📂 Demo：[LLM_aBatch.py](./day02_models_io/part03_prompt/invoke/LLM_aBatch.py)

```python
async def async_batch_call():
    response = await model.abatch(questions)   # 异步批量
    for q, r in zip(questions, response):
        print(f"问题：{q}\n回答：{r.content}\n")

asyncio.run(async_batch_call())
```


### 3.6 stream — 同步流式调用
> 📂 Demo：[LLM_Stream.py](./day02_models_io/part03_prompt/invoke/LLM_Stream.py)

```python
messages = [
    SystemMessage(content="你叫小问，是一个乐于助人的AI人工助手"),
    HumanMessage(content="你是谁")
]

response = model.stream(messages)    # 返回 Iterator

for chunk in response:
    print(chunk.content, end="", flush=True)  # flush=True 立即刷新缓冲区
```


### 3.7 astream — 异步流式调用
> 📂 Demo：[LLM_aStream.py](./day02_models_io/part03_prompt/invoke/LLM_aStream.py)

```python
async def async_stream_call():
    response = model.astream(messages)   # 返回 async_generator

    async for chunk in response:         # 必须用 async for
        print(chunk.content, end="", flush=True)

asyncio.run(async_stream_call())
```


### 3.8 同步 vs 异步 性能对比
> 📂 Demo：[sync_vs_async_demo.py](./day02_models_io/part03_prompt/invoke/sync_vs_async_demo.py)

| 方案 | 写法 | 2个任务(2s+3s)耗时 | 原理 |
|------|------|---------------------|------|
| 同步 `invoke` | 顺序调用 | ~5 秒 | 阻塞等待，逐个执行 |
| 异步顺序 `await` | `await a`; `await b` | ~5 秒 | 写法异步，但仍是逐个等 |
| 异步并发 `gather` | `asyncio.gather(a, b)` | ~3 秒 | 两个任务同时跑，取最长耗时 |

```python
# 真正的异步并发
async def demo_async_concurrent():
    task1 = asyncio.create_task(model.ainvoke("问题1"))
    task2 = asyncio.create_task(model.ainvoke("问题2"))
    r1, r2 = await asyncio.gather(task1, task2)
    # 总耗时 ≈ max(t1, t2)，而非 t1 + t2
```


---

## 四、消息类型（Messages）

### 4.1 五种消息类型
> 📂 Demo：[ChatPromptTemplate_MessageParam.py](./day02_models_io/part03_prompt/chat_prompt_template/parameter/ChatPromptTemplate_MessageParam.py)

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="你好，请你介绍一下你自己"),
    AIMessage(content="我是一名人工智能助手，请问您有什么想问的嘛?"),
    ToolMessage(
        tool_call_id="call_abc123",    # 关联的工具调用 ID
        content='{"population": 21540000, "area": "16410平方公里"}',
    )
]
```

| 消息类型 | 角色 | 说明 |
|----------|------|------|
| `SystemMessage` | system | 设定 AI 的行为、角色、规则 |
| `HumanMessage` | user | 用户的输入 |
| `AIMessage` | assistant | AI 的回复 |
| `ToolMessage` | tool | 工具调用返回的结果，需关联 `tool_call_id` |
| `BaseMessage` | — | 所有消息的父类 |


### 4.2 消息对象 vs 字典

```python
# 方式1：使用消息对象（推荐，类型安全）
from langchain_core.messages import SystemMessage, HumanMessage
messages = [
    SystemMessage(content="你是AI助手"),
    HumanMessage(content="请问：{question}")
]

# 方式2：使用字典（兼容 OpenAI 格式）
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
]
```

---

## 五、PromptTemplate（提示词模板）

> `PromptTemplate` 用于**单条字符串**模板，适合简单的一问一答场景。

### 5.1 构造方法（两种方式）
> 📂 Demo：[PromptTemplate_Constructor.py](./day02_models_io/part03_prompt/prompt_templates/PromptTemplate_Constructor.py) | [PromptTemplate_FromTemplate.py](./day02_models_io/part03_prompt/prompt_templates/PromptTemplate_FromTemplate.py)

```python
from langchain_core.prompts import PromptTemplate

# 方式1：构造函数
template = PromptTemplate(
    template="你是一个专业的{role}工程师，请回答：{question}",
    input_variables=['role', 'question']
)

# 方式2：from_template（更简洁）
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答：{question}"
)
```

| 参数 | 说明 |
|------|------|
| `template` | 模板字符串，用 `{变量名}` 做占位符 |
| `input_variables` | 变量名列表，声明模板中的所有变量 |


### 5.2 format() — 填充变量（返回字符串）

```python
prompt = template.format(role="python开发", question="冒泡排序怎么写？")
# 返回: "你是一个专业的python开发工程师，请回答：冒泡排序怎么写？"
```

### 5.3 invoke() — 填充变量（返回 PromptValue 对象）
> 📂 Demo：[PromptTemplate_FormatMethod.py](./day02_models_io/part03_prompt/prompt_templates/method/PromptTemplate_FormatMethod.py) | [PromptTemplate_InvokeMethod.py](./day02_models_io/part03_prompt/prompt_templates/method/PromptTemplate_InvokeMethod.py)

```python
prompt_value = template.invoke({"role": "python开发", "question": "冒泡排序怎么写？"})

# 转为字符串
print(prompt_value.to_string())

# 转为消息列表
print(prompt_value.to_messages())
```


| 方法 | 返回值 | 转换方式 |
|------|--------|----------|
| `format(**kwargs)` | `str` | 直接是字符串 |
| `invoke(dict)` | `PromptValue` | `.to_string()` 或 `.to_messages()` |

> **区别**：`format()` 返回纯字符串，`invoke()` 返回 `PromptValue` 对象（是 LCEL Runnable 体系的标准入口）。

### 5.4 partial() / partial_variables — 部分变量绑定
> 📂 Demo：[PromptTemplate_PartialMethod.py](./day02_models_io/part03_prompt/prompt_templates/method/PromptTemplate_PartialMethod.py) | [PromptTemplate_PartialVariables.py](./day02_models_io/part03_prompt/prompt_templates/PromptTemplate_PartialVariables.py)

```python
from datetime import datetime

# 方式1：构造时预设 partial_variables
template1 = PromptTemplate.from_template(
    "现在时间是：{time}，请回答：{question}",
    partial_variables={"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
)
prompt1 = template1.format(question="今天是几号？")

# 方式2：调用 partial() 方法
template2 = PromptTemplate.from_template(
    "现在时间是：{time}，请回答：{question}"
)
partial = template2.partial(time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
prompt2 = partial.format(question="今天是几号？")
```

| API | 说明 |
|-----|------|
| `partial_variables={...}` | 构造时预设部分变量的值，后续只填剩余变量 |
| `.partial(key=value)` | 返回新模板，已绑定指定变量 |
| `format()` 传入同名变量 | 可**覆盖** `partial_variables` 中预设的值 |


### 5.5 模板拼接
> 📂 Demo：[PromptTemplate_Combined.py](./day02_models_io/part03_prompt/prompt_templates/PromptTemplate_Combined.py)

```python
prompt_a = PromptTemplate.from_template("请用一句话介绍{topic}，要求通俗易懂\n")
prompt_b = PromptTemplate.from_template("内容不超过{length}个字")

prompt_all = prompt_a + prompt_b   # 两个模板拼接
result = prompt_all.format(topic="LangChain", length=200)
```

> **用途**：在 AI 产品中分段构建复杂 Prompt，多组件一言一语组合成最终提示词。


---

## 六、ChatPromptTemplate（聊天提示词模板）

> `ChatPromptTemplate` 用于**多角色消息列表**，适合系统角色设定 + 多轮对话场景。

### 6.1 构造方法
> 📂 Demo：[ChatPromptTemplate_Constructor.py](./day02_models_io/part03_prompt/chat_prompt_template/ChatPromptTemplate_Constructor.py) | [ChatPromptTemplate_TupleParam.py](./day02_models_io/part03_prompt/chat_prompt_template/parameter/ChatPromptTemplate_TupleParam.py) | [ChatPromptTemplate_DictParam.py](./day02_models_io/part03_prompt/chat_prompt_template/parameter/ChatPromptTemplate_DictParam.py)

```python
from langchain_core.prompts import ChatPromptTemplate

# 构造时传入消息列表
chatPromptTemplate = ChatPromptTemplate(
    [
        ("system", "你是一个AI开发工程师，你的名字是{name}。"),
        ("human", "你能帮我做什么?"),
        ("ai", "我能开发很多{thing}。"),
        ("human", "{user_input}"),
    ]
)
```

**messages 参数支持三种格式：**

```python
# 格式1：tuple 列表 [(role, content)]
[
    ("system", "你是AI助手，你的名字叫{name}。"),
    ("user", "请问：{question}")
]

# 格式2：dict 列表 [{"role":..., "content":...}]
[
    {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
    {"role": "user", "content": "请问：{question}"}
]

# 格式3：Message 类列表
from langchain_core.messages import SystemMessage, HumanMessage
[
    SystemMessage(content="你是AI助手，你的名字叫{name}。"),
    HumanMessage(content="请问：{question}")
]
```


### 6.2 from_messages() — 工厂方法

```python
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，请回答我提出的问题"),
        ("human", "请回答:{question}")
    ]
)
```

### 6.3 三种填充方法对比
> 📂 Demo：[ChatPromptTemplate_FormatMessages.py](./day02_models_io/part03_prompt/chat_prompt_template/ChatPromptTemplate_FormatMessages.py)

```python
# 1. format_messages() — 返回 List[BaseMessage]
messages = chat_prompt.format_messages(role="python开发工程师", question="堆排序怎么写")
# ➜ [SystemMessage(...), HumanMessage(...)]

# 2. invoke() — 返回 PromptValue（LCEL 标准入口）
prompt_value = chat_prompt.invoke({"role": "python开发工程师", "question": "堆排序怎么写"})
print(prompt_value.to_string())    # 转为纯文本
print(prompt_value.to_messages())  # 转为消息列表

# 3. format() — 返回纯字符串
text = chat_prompt.format(**{"role": "python开发工程师", "question": "快速排序怎么写"})
```

| 方法 | 返回类型 | 适用场景 |
|------|----------|----------|
| `format_messages(**kwargs)` | `list[BaseMessage]` | 直接传给模型调用 |
| `invoke(dict)` | `PromptValue` | LCEL Chain 中串联 |
| `format(**kwargs)` | `str` | 调试 / 查看最终文本 |


### 6.4 MessagesPlaceholder — 历史消息占位符
> 📂 Demo：[ChatPromptTemplate_ExplicitPlaceholder.py](./day02_models_io/part03_prompt/chat_prompt_template/placeholder/ChatPromptTemplate_ExplicitPlaceholder.py) | [ChatPromptTemplate_ImplicitPlaceholder.py](./day02_models_io/part03_prompt/chat_prompt_template/placeholder/ChatPromptTemplate_ImplicitPlaceholder.py)

**场景**：多轮对话中，需要把历史聊天记录插入模板。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 显式使用 MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的Python应用开发工程师"),
    MessagesPlaceholder("memory"),          # 显式占位
    ("human", "{question}")
])

# 隐式写法（简写）
prompt = ChatPromptTemplate.from_messages([
    ("placeholder", "{memory}"),            # 等价于 MessagesPlaceholder("memory")
    ("system", "你是一个资深的Python应用开发工程师"),
    ("human", "{question}")
])

# 调用时传入历史消息
prompt_value = prompt.invoke({
    "memory": [
        HumanMessage("我的名字叫亮仔，是一名程序员"),
        AIMessage("好的，亮仔你好")
    ],
    "question": "请问我的名字叫什么？"
})
```

| 写法 | 说明 |
|------|------|
| `MessagesPlaceholder("memory")` | **显式**占位，推荐（更清晰） |
| `("placeholder", "{memory}")` | **隐式**简写，等价 |


---

## 七、外部加载 Prompt

### 7.1 load_prompt() — 从文件加载
> 📂 Demo：[PromptLoadDemo01.py](./day02_models_io/part03_prompt/load_external/PromptLoadDemo01.py) | [PromptLoadDemo02.py](./day02_models_io/part03_prompt/load_external/PromptLoadDemo02.py)

```python
from langchain_core.prompts import load_prompt

# 从 JSON 加载
template = load_prompt("prompt.json", encoding="utf-8")

# 从 YAML 加载
template = load_prompt("prompt.yaml", encoding="utf-8")

# 使用
print(template.format(name="张三", what="搞笑的"))
```

**JSON 格式示例（prompt.json）：**

```json
{
    "input_variables": ["name", "what"],
    "template": "请{name}讲一个{what}的故事"
}
```

**YAML 格式示例（prompt.yaml）：**

```yaml
input_variables: ["name", "what"]
template: "请{name}讲一个{what}的故事"
```

> **用途**：Prompt 工程化管理，模板与代码分离，方便非开发人员维护提示词。


---

## 八、输出解析器（Output Parser）

> 输出解析器将 LLM 的自由文本输出转换为结构化数据。

### 8.1 StrOutputParser — 字符串解析器
> 📂 Demo：[StrOutputParserDemo.py](./day03_parser/StrOutputParserDemo.py)

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
response = parser.invoke(result)   # 提取 AIMessage 的 .content 字段
# 返回类型: str
```

> **作用**：最简单解析器，从 `AIMessage` 中提取 `.content` 字符串。


### 8.2 JsonOutputParser — JSON 解析器

#### 基础用法（提示词中手动约束 JSON）
> 📂 Demo：[JsonOutputParserDemo.py](./day03_parser/JsonOutputParserDemo.py)

```python
from langchain_core.output_parsers import JsonOutputParser

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，结果返回json格式，q字段表示问题，a字段表示答案。"),
    ("human", "请回答:{question}")
])

result = model.invoke(prompt)

parser = JsonOutputParser()
response = parser.invoke(result)   # 返回类型: dict
# 如: {"q": "什么是LangChain", "a": "LangChain是..."}
```

> **局限**：依赖提示词引导模型输出 JSON，模型可能不遵守格式，解析失败。

#### 进阶用法（get_format_instructions + Pydantic）
> 📂 Demo：[JsonOutputParserDemo.py](./day03_parser/JsonOutputParserDemo.py) | [JsonOutputParser_GetFormatInstructions.py](./day03_parser/JsonOutputParser_GetFormatInstructions.py)

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class Person(BaseModel):
    time: str = Field(description="时间")
    person: str = Field(description="人物")
    event: str = Field(description="事件")

parser = JsonOutputParser(pydantic_object=Person)

# 自动生成格式指令
format_instructions = parser.get_format_instructions()
# 输出类似: "The output should be a JSON object with the following fields: time, person, event..."

# 将格式指令嵌入提示词
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化JSON数据。"),
    ("human", "请生成一个关于{topic}的新闻。{format_instructions}")
])

prompt = chat_prompt.format_messages(topic="小米su7跑车", format_instructions=format_instructions)
result = model.invoke(prompt)
response = parser.invoke(result)   # 返回 Person 对象
```


### 8.3 PydanticOutputParser — Pydantic 输出解析器
> 📂 Demo：[JsonOutputParser_GetFormatInstructions.py](./day03_parser/JsonOutputParser_GetFormatInstructions.py)（含 PydanticOutputParser 示例）

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

class Product(BaseModel):
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    description: str = Field(description="产品简介")

    @field_validator("description")
    def validate_description(cls, value):
        """自定义校验：描述至少10个字符"""
        if len(value) < 10:
            raise ValueError('产品简介长度必须大于等于10')
        return value

parser = PydanticOutputParser(pydantic_object=Product)
format_instructions = parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手，你只能输出结构化的json数据\n{format_instructions}"),
    ("human", "请你输出标题为：{topic}的新闻内容")
])

prompt = prompt_template.format_messages(topic="华为Mate X7", format_instructions=format_instructions)
result = model.invoke(prompt)
response = parser.invoke(result)   # 返回 Product 对象，自动校验
```

| 特性 | `JsonOutputParser` | `PydanticOutputParser` |
|------|--------------------|-------------------------|
| 基础功能 | 解析为 dict | 解析为 Pydantic 对象 |
| 类型校验 | ❌ | ✅ 自动校验字段类型 |
| 自定义校验器 | ❌ | ✅ `@field_validator` |
| 默认值/可选字段 | ❌ | ✅ Pydantic 原生支持 |
| 适用场景 | 简单 JSON | 复杂结构 + 强类型约束 |


---

## 九、结构化输出（with_structured_output）

> ⭐ **推荐方式**：在 API 层面强制 JSON 输出，比提示词引导更可靠。

### 9.1 对比旧方式

| | 旧方式（JsonOutputParser） | 新方式（with_structured_output） |
|------|---------------------------|----------------------------------|
| 机制 | 提示词描述格式 | API 层面约束 |
| 稳定性 | 依赖模型遵守 | 强制合法 JSON |
| 额外 parser | 需要 | 不需要 |
| 返回值 | 需手动解析 | 直接是 Python 对象 |

### 9.2 使用 Pydantic BaseModel（带校验）
> 📂 Demo：[StructuredOutput_Demo.py](./day03_parser/StructuredOutput_Demo.py) | [StructuredOutput_Pydantic.py](./day03_parser/StructuredOutput_Pydantic.py)

```python
from pydantic import BaseModel, Field

class News(BaseModel):
    """新闻结构化模型"""
    title: str = Field(description="新闻标题")
    time: str = Field(description="新闻发生的时间")
    person: str = Field(description="新闻涉及的人物")
    event: str = Field(description="发生的具体事件")

# 绑定结构化输出
news_model = model.with_structured_output(News)

# 直接调用，返回 News 对象
result = news_model.invoke([{"role": "user", "content": "请生成一条关于小米SU7跑车发布的新闻"}])

print(result.title)    # 直接访问属性
print(result.time)
print(result.person)
print(result.event)
# 返回类型: News (Pydantic BaseModel)
```


### 9.3 使用 TypedDict（轻量）
> 📂 Demo：[StructuredOutput_TypedDict.py](./day03_parser/StructuredOutput_TypedDict.py)

```python
from typing import TypedDict, Annotated

class Weather(TypedDict):
    city: Annotated[str, "城市名称"]
    temperature: Annotated[str, "温度，如25°C"]
    condition: Annotated[str, "天气状况，如晴、多云、雨"]

weather_model = model.with_structured_output(Weather)
result = weather_model.invoke([{"role": "user", "content": "查询北京天气"}])

print(result['city'])         # TypedDict 返回 dict 类
print(result['temperature'])
```

### 9.4 嵌套结构（TypedDict）
> 📂 Demo：[StructuredOutput_TypedDict.py](./day03_parser/StructuredOutput_TypedDict.py)

```python
class Animal(TypedDict):
    animal: Annotated[str, "动物"]
    emoji: Annotated[str, "表情"]

class AnimalList(TypedDict):
    animals: Annotated[list[Animal], "动物与表情列表"]

llm_with_structured_output = llm.with_structured_output(AnimalList)
resp = llm_with_structured_output.invoke(
    [{"role": "user", "content": "任意生成三种动物，以及他们的 emoji 表情"}]
)
# 返回: {'animals': [{'animal': '猫', 'emoji': '🐱'}, ...]}
```


### 9.5 Pydantic vs TypedDict 选择建议

| 场景 | 推荐 |
|------|------|
| 需要字段校验（范围、长度等） | `Pydantic BaseModel` |
| 简单数据结构 | `TypedDict` |
| 需要默认值 | `Pydantic BaseModel` |
| 轻量无依赖 | `TypedDict` |

---

## 十、类型注解（Annotated + Pydantic / TypedDict）

### 10.1 Annotated + Pydantic Field（有运行时校验 ✅）
> 📂 Demo：[AnnotatedPydantic.py](./day03_parser/AnnotatedPydantic.py)

```python
from typing import Annotated
from pydantic import BaseModel, Field, ValidationError

Age = Annotated[int, Field(ge=0, le=150, description="年龄，范围0-150")]

class Person(BaseModel):
    name: str
    age: int
    age2: Age

try:
    p = Person(name="z3", age=11, age2=188)   # age2 > 150
except ValidationError as e:
    print("数据校验失败：", e)                  # ✅ 运行时校验生效
```


### 10.2 Annotated + TypedDict（无运行时校验 ❌）
> 📂 Demo：[AnnotatedTypedDict.py](./day03_parser/AnnotatedTypedDict.py)

```python
from typing import Annotated, TypedDict

Age = Annotated[int, "年龄，范围0-150"]

class Person(TypedDict):
    name: str
    age: int
    age2: Age

p = Person(name="z3", age=111, age2=188)   # 不会报错，188 > 150 也能通过
# Annotated 在 TypedDict 中仅是元数据（文档说明），不做运行时校验
```

| 组合 | 运行时校验 | 用途 |
|------|------------|------|
| `Annotated[int, Field(ge=0, le=150)]` + Pydantic | ✅ | 强类型校验 |
| `Annotated[int, "年龄"]` + TypedDict | ❌ | 仅文档注释 |

> **核心原因**：`Annotated` 的设计目的是为类型添加元数据，而非运行时校验。Pydantic 的 `Field` 自带校验逻辑，TypedDict 只是静态类型提示。


---

## 附录：完整导入速查

```python
# ==================== 环境 ====================
from dotenv import load_dotenv
import os

# ==================== 模型初始化 ====================
from langchain.chat_models import init_chat_model           # v1.0+ 统一入口
from langchain_openai import ChatOpenAI                     # OpenAI 兼容
from langchain_deepseek import ChatDeepSeek                 # DeepSeek 专用
from langchain_ollama import ChatOllama                     # Ollama 本地
from langchain_community.chat_models.tongyi import ChatTongyi # 通义千问
from langchain_qwq import ChatQwen                          # 通义千问新版
from langchain_anthropic import ChatAnthropic               # Anthropic Claude

# ==================== 消息类型 ====================
from langchain.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)

# ==================== Prompt 模板 ====================
from langchain_core.prompts import (
    PromptTemplate,            # 字符串模板
    ChatPromptTemplate,        # 聊天模板
    MessagesPlaceholder,       # 历史消息占位符
    load_prompt,               # 从文件加载
)

# ==================== 输出解析器 ====================
from langchain_core.output_parsers import (
    StrOutputParser,           # 字符串解析
    JsonOutputParser,          # JSON 解析
    PydanticOutputParser,      # Pydantic 解析（带校验）
)

# ==================== 数据模型 ====================
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import TypedDict, Annotated

# ==================== 调用方式 ====================
# model.invoke(input)          同步单次
# model.ainvoke(input)         异步单次
# model.batch(inputs)          同步批量
# model.abatch(inputs)         异步批量
# model.stream(input)          同步流式
# model.astream(input)         异步流式
# model.with_structured_output(Schema)   结构化输出
```

> 📅 文档生成时间：2026-07-21 — 基于尚硅谷 LangChain Day01~Day03 课程内容整理