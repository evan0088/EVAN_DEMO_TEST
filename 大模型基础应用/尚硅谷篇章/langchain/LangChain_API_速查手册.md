# LangChain 六天课程 — API 速查手册

> 涵盖 Day01（入门）→ Day02（模型 I/O + Prompt）→ Day03（输出解析器）→ Day04（LCEL + Memory + Tools）→ **Day05（Embedding + 向量数据库 + RAG）** → **Day06（MCP 协议 + Agent）** 全部 API

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
- [十一、LCEL（LangChain Expression Language）](#十一lcellangchain-expression-language)
- [十二、Memory（对话记忆）](#十二memory对话记忆)
- [十三、Tools（工具定义与调用）](#十三tools工具定义与调用)
- [十四、Embedding（文本向量化）](#十四embedding文本向量化)
- [十五、向量数据库 Redis](#十五向量数据库-redis)
- [十六、文档加载器（Document Loaders）](#十六文档加载器document-loaders)
- [十七、文本分割器（Text Splitters）](#十七文本分割器text-splitters)
- [十八、RAG 完整管道](#十八rag-完整管道)
- [十九、MCP 协议](#十九mcp-协议)
- [二十、Agent（智能体）](#二十agent智能体)
- [二十一、Agent 高级模式](#二十一agent-高级模式)
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
    api_key=os.getenv("DEEPSEEK_API_KEY"),
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
    api_key=os.getenv("DEEPSEEK_API_KEY"),
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

## 十一、LCEL（LangChain Expression Language）

> LCEL 是 LangChain 的声明式链式编程语言，通过 `|` 管道符将组件串联，让一个任务的输出成为下一个任务的输入。

### 11.1 LCEL 概述 — 管道符 `|`

LCEL 的核心是 `|`（管道符），等价于 Linux 管道：`prompt | model | parser` 表示 prompt 的输出 → model 的输入 → parser 的输入。

**LCEL 的优势：**
- **简洁**：一行链式调用代替多行代码
- **异步 / 流式 / 批量**：所有 Runnable 统一支持 `invoke` / `ainvoke` / `stream` / `batch`
- **可组合**：任意组合 prompt、model、parser、lambda、branch 等
- **可观测**：支持 `get_graph().print_ascii()` 可视化链结构

### 11.2 RunnableSequence — 顺序链（串行）

> 📂 Demo：[LCEL_RunnableSequenceDemo.py](./day04/06_lcel/LCEL_RunnableSequenceDemo.py) | [LCEL_RunnableSerializableDemo.py](./day04/06_lcel/LCEL_RunnableSerializableDemo.py)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 三个独立组件
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题"),
    ("human", "请回答:{question}")
])
model = init_chat_model(...)
parser = StrOutputParser()

# LCEL 串联（管道符）
chain = chat_prompt | model | parser

# 一行调用
result = chain.invoke({"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"})
# 返回类型: str
```

**多步骤串联（子链叠加）：**

```python
# 子链1：生成中文内容
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
    ("human", "请简短介绍什么是{topic}")
])
chain1 = prompt1 | model | StrOutputParser()

# 子链2：翻译成英文
prompt2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译助手，将用户输入内容翻译成英文"),
    ("human", "{input}")
])
chain2 = prompt2 | model | StrOutputParser()

# 串联两个子链：chain1 的输出通过 lambda 传给 chain2
full_chain = chain1 | (lambda content: {"input": content}) | chain2

result = full_chain.invoke({"topic": "langchain"})
# chain1 输出中文介绍 → lambda 包装为 {"input": ...} → chain2 翻译为英文
```

| API | 说明 |
|-----|------|
| `A \| B` | 管道符，A 的输出成为 B 的输入 |
| `chain1 \| (lambda x: {...}) \| chain2` | 用 lambda 做数据转换，将上游输出 reshape 为下游需要的 dict |

> **注意**：`chain1` 输出是纯字符串，`chain2` 需要 `{"input": "..."}` 格式的 dict，因此必须用 lambda 做中间转换。


### 11.3 RunnableParallel — 并行链

> 📂 Demo：[LCEL_RunnableParallelDemo.py](./day04/06_lcel/LCEL_RunnableParallelDemo.py)

**作用**：同时执行多个 Runnable，合并结果。

```python
from langchain_core.runnables import RunnableParallel

# 中文链
chain_cn = prompt_cn | model | parser_cn

# 英文链
chain_en = prompt_en | model | parser_en

# 并行执行
parallel_chain = RunnableParallel({
    "chinese": chain_cn,
    "english": chain_en
})

result = parallel_chain.invoke({"topic": "langchain"})
# 返回: {"chinese": "LangChain是...", "english": "LangChain is..."}
```

| API | 说明 |
|-----|------|
| `RunnableParallel({key: runnable, ...})` | 并行执行多个 Runnable，返回 dict |
| `parallel_chain.invoke(input)` | 将同一个 input 分发给所有子链 |
| `.get_graph().print_ascii()` | 打印链的 ASCII 可视化结构 |

> **用途**：一次请求同时获取多种视角结果（如中英文翻译、多维度分析），总耗时 ≈ 最慢子链耗时。


### 11.4 RunnableLambda — 函数链

> 📂 Demo：[LCEL_RunnableLambdaDemo.py](./day04/06_lcel/LCEL_RunnableLambdaDemo.py)

**作用**：将普通 Python 函数包装为 Runnable，融入 LCEL 链中。

```python
from langchain_core.runnables import RunnableLambda

# 普通 Python 函数
def debug_print(x):
    """调试用的中间打印函数"""
    logger.info(f"中间结果: {x}")
    return {"input": x}

# 包装为 Runnable
debug_node = RunnableLambda(debug_print)

# 嵌入 LCEL 链中做调试
full_chain = chain1 | debug_node | chain2

result = full_chain.invoke({"topic": "langchain"})
```

| API | 说明 |
|-----|------|
| `RunnableLambda(func)` | 将普通函数转为 Runnable，融入管道链 |
| 函数签名 | 接收上游输出，返回下游需要的格式 |

**常见用法：**

| 场景 | 示例 |
|------|------|
| 调试打印 | `RunnableLambda(lambda x: print(f"中间结果: {x}") or x)` |
| 数据转换 | `(lambda content: {"input": content})` |
| 结果记录 | 将中间结果写入日志/数据库 |


### 11.5 RunnableBranch — 分支链

> 📂 Demo：[LCEL_RunnableBranchDemo.py](./day04/06_lcel/LCEL_RunnableBranchDemo.py)

**作用**：根据输入动态选择不同的处理路径（if-else 逻辑）。

```python
from langchain_core.runnables import RunnableBranch

# 定义不同分支的 prompt
english_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个英语翻译专家，你叫小英"),
    ("human", "{query}")
])
japanese_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个日语翻译专家，你叫小日"),
    ("human", "{query}")
])
korean_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个韩语翻译专家，你叫小韩"),
    ("human", "{query}")
])

# 路由判断函数
def determine_language(inputs):
    query = inputs["query"]
    if "日语" in query:
        return "japanese"
    elif "韩语" in query:
        return "korean"
    else:
        return "english"

# 构建分支链
chain = RunnableBranch(
    (lambda x: determine_language(x) == "japanese", japanese_prompt | model | parser),
    (lambda x: determine_language(x) == "korean", korean_prompt | model | parser),
    (english_prompt | model | parser)  # 默认分支（无条件，最后兜底）
)

# 自动路由到对应分支
result = chain.invoke({"query": '请你用韩语翻译这句话:"见到你很高兴"'})
```

| API | 说明 |
|-----|------|
| `RunnableBranch((条件1, 分支1), (条件2, 分支2), ..., 默认分支)` | 按顺序匹配条件，命中则执行对应分支 |
| 条件函数 | `(lambda x: bool)` 或任意返回 `bool` 的函数 |
| 默认分支 | 不带条件的最后一项，无条件执行（兜底） |

> **等价于**：`if ... elif ... else` 逻辑，但保持 LCEL 的 Runnable 类型一致性，支持串流/异步/批量等所有 Runnable 能力。


### 11.6 LCEL 链可视化

```python
# 查看链的结构（ASCII 图）
chain.get_graph().print_ascii()

# 生成 Mermaid 图
print(chain.get_graph().draw_mermaid())
```

> **用途**：调试复杂链结构，直观查看数据流转路径。


---

## 十二、Memory（对话记忆）

> LLM 每次调用是无状态的——模型本身不记得上一轮对话。Memory 模块通过**外部存储历史消息**并在每次请求时注入上下文，实现"记忆"效果。

### 12.1 问题的本质：LLM 的"遗忘"

> 📂 Demo：[Memory_IDontKnow.py](./day04/07_memory/Memory_IDontKnow.py)

```python
chain = prompt | llm | parser

# 第一轮对话
print(chain.invoke({"question": "我叫张三，你叫什么?"}))
# 输出: "你好张三，我叫通义千问..."

# 第二轮对话
print(chain.invoke({"question": "你知道我是谁吗?"}))
# 输出: "我不知道你是谁..."  ← 忘了！
```

> **根本原因**：每次 `invoke()` 是独立的无状态 HTTP 请求，LLM 没有"记忆"，必须把历史对话随每次请求一起发送。


### 12.2 InMemoryChatMessageHistory — 内存聊天历史

> 📂 Demo：[Memory_InMemoryChatMessageHistory.py](./day04/07_memory/Memory_InMemoryChatMessageHistory.py)

**手动管理历史记录（理解原理）：**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()

# 添加消息
history.add_user_message("我叫张三，我的爱好是学习")
ai_message = llm.invoke(history.messages)   # 传入历史消息
history.add_message(ai_message)              # 记录 AI 回复

# 新一轮对话（携带历史）
history.add_user_message("我叫什么？我的爱好是什么？")
ai_message2 = llm.invoke(history.messages)   # 模型能看到之前的内容

# 遍历全部消息
for message in history.messages:
    print(message.content)
```

| API | 说明 |
|-----|------|
| `InMemoryChatMessageHistory()` | 创建内存中的聊天历史实例 |
| `.add_user_message(content)` | 添加一条用户消息 |
| `.add_message(ai_message)` | 添加一条 AI 消息 |
| `.messages` | 返回完整的消息列表 `list[BaseMessage]` |
| `.clear()` | 清空历史 |

> **局限**：进程重启后数据丢失，无法多会话隔离。


### 12.3 RunnableWithMessageHistory — 可持续记忆（基础版）

> 📂 Demo：[Memory_RunnableWithMessageHistory.py](./day04/07_memory/Memory_RunnableWithMessageHistory.py)

**自动管理历史 + 链式调用：**

```python
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory

# Prompt 中预留历史消息槽位
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),   # 历史消息注入点
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# 创建内存历史实例
history = InMemoryChatMessageHistory()

# 包装为带记忆的链
runnable = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda session_id: history,  # 获取历史的函数
    input_messages_key="input",       # 对应 prompt 中用户输入的 key
    history_messages_key="history"    # 对应 MessagesPlaceholder 的变量名
)

config = RunnableConfig(configurable={"session_id": "user-001"})

# 调用时多轮对话自动带历史
runnable.invoke({"input": "我叫张三，我爱好学习。"}, config)
runnable.invoke({"input": "我叫什么？我的爱好是什么？"}, config)
# 第二句能正确回答！
```

| API | 说明 |
|-----|------|
| `RunnableWithMessageHistory(chain, get_session_history, input_messages_key, history_messages_key)` | 包装链，自动管理历史消息 |
| `RunnableConfig(configurable={"session_id": "..."})` | 运行时配置，用于区分不同会话 |
| `get_session_history` | 接收 `session_id` 返回对应历史对象的函数 |

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `chain` | `Runnable` | 被包装的原始链 |
| `get_session_history` | `Callable[[str], BaseChatMessageHistory]` | 根据 session_id 返回历史对象 |
| `input_messages_key` | `str` | 输入 dict 中用户消息的键名 |
| `history_messages_key` | `str` | 对应 `MessagesPlaceholder` 的变量名 |


### 12.4 多会话管理（进阶版）

> 📂 Demo：[Memory_RunnableWithMessageHistoryV2.py](./day04/07_memory/Memory_RunnableWithMessageHistoryV2.py)

**用 dict 存储多个 session 的历史，实现多用户隔离：**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 全局会话存储
store = {}

def get_session_history(session_id: str):
    """按 session_id 获取或创建历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的中文助理，会根据上下文回答问题。"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

chain = prompt | llm | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 不同 session_id = 独立会话
cfg_user1 = {"configurable": {"session_id": "user-001"}}
cfg_user2 = {"configurable": {"session_id": "user-002"}}

print(with_history.invoke({"question": "我叫张三。"}, cfg_user1))
# user-001: "你好张三！"
print(with_history.invoke({"question": "我叫李四。"}, cfg_user2))
# user-002: "你好李四！"  ← 独立会话，互不干扰
print(with_history.invoke({"question": "我叫什么？"}, cfg_user1))
# user-001: "你叫张三。" ← 各自记住自己的上下文
```

| 场景 | 说明 |
|------|------|
| 单用户 | `lambda session_id: history`（永远返回同一对象） |
| 多用户 | `dict[session_id]` 模式（不同 session 各自独立） |
| 生产环境 | 改用 Redis / SQLite 等持久化存储 |

> **核心思想**：`store` 就是"会话池"，key = `session_id`，value = 该会话的消息历史。


### 12.5 RedisChatMessageHistory — Redis 持久化记忆

> 📂 Demo：[Memory_RedisChatMessageHistory.py](./day04/07_memory/Memory_RedisChatMessageHistory.py) | [RedisEnvCheck.py](./day04/07_memory/RedisEnvCheck.py)

**生产环境：用 Redis 持久化会话历史。**

```bash
pip install redis==5.3.1
```

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import redis

REDIS_URL = "redis://localhost:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
        # ttl=3600  # 可选：设置过期时间（秒）
    )

chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = RunnableConfig(configurable={"session_id": "user-001"})

# 交互式对话循环
while True:
    question = input("\n输入问题：")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    response = chain.invoke({"question": question}, config)
    print(f"AI: {response.content}")
    redis_client.save()   # 强制持久化到 dump.rdb
```

| API | 说明 |
|-----|------|
| `RedisChatMessageHistory(session_id, url)` | 创建 Redis 会话历史对象 |
| `RedisChatMessageHistory(session_id, url, ttl=3600)` | 设置过期时间（秒），过期自动清理 |
| `redis.Redis.from_url(url, decode_responses=True)` | 创建原生 Redis 客户端 |
| `redis_client.save()` | 强制写入 dump.rdb（等同于 `redis-cli SAVE`） |

> **注意**：`decode_responses=True` 让 Redis 返回字符串而非字节串。


### 12.6 RunnableConfig — 会话配置

```python
from langchain_core.runnables import RunnableConfig

# 完整写法
config = RunnableConfig(configurable={"session_id": "user-001"})

# 简写（dict 自动转换）
config = {"configurable": {"session_id": "user-001"}}
```

| 参数 | 说明 |
|------|------|
| `configurable` | 可配置的 dict，`session_id` 为约定键名 |
| `session_id` | 会话标识，相当于"登录用户名"——不同 ID 对应不同历史 |

### 12.7 记忆方案对比

| 方案 | 持久化 | 多会话 | 适用场景 |
|------|--------|--------|----------|
| `InMemoryChatMessageHistory` | ❌ | ❌（单对象） | 原型开发、测试 |
| `store = {}` + InMemory | ❌ | ✅ | 本地多用户模拟 |
| `RedisChatMessageHistory` | ✅ | ✅ | 生产环境 |
| SQLite / Postgres | ✅ | ✅ | 需要 SQL 查询历史 |

---

## 十三、Tools（工具定义与调用）

> 工具（Tool）让 LLM 能够调用外部函数——查询数据库、调用 API、执行计算等，突破纯文本生成的局限。

### 13.1 @tool 装饰器 — 基础用法

> 📂 Demo：[Tool_AddNumberTool.py](./day04/08_tools/Tool_AddNumberTool.py)

**将普通 Python 函数注册为 LangChain 工具：**

```python
from langchain.tools import tool

@tool
def add_number(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

# 调用工具（必须用 .invoke(dict)）
result = add_number.invoke({"a": 1, "b": 12})
# 返回: 13

# 工具元信息
print(add_number.name)         # 'add_number'
print(add_number.description)  # '两个整数相加'
print(add_number.args)         # {'a': {'title': 'A', 'type': 'integer'}, 'b': ...}
```

| 属性 | 说明 |
|------|------|
| `.name` | 工具名称，默认使用函数名 |
| `.description` | 工具描述，取自函数 docstring（**必须提供**） |
| `.args` | 工具参数 schema，从类型注解自动推导 |
| `.invoke(dict)` | 调用工具，传入参数字典 |

> **关键约定**：`@tool` 装饰器用函数 **docstring** 作为工具描述，LLM 据此判断何时调用该工具——docstring 必须清晰准确。


### 13.2 @tool 装饰器 — 进阶：args_schema 自定义参数

> 📂 Demo：[Tool_AddNumberToolPro.py](./day04/08_tools/Tool_AddNumberToolPro.py)

**使用 Pydantic BaseModel 精确定义工具参数：**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class FieldInfo(BaseModel):
    """定义加法运算所需的参数信息"""
    a: int = Field(description="第1个参数")
    b: int = Field(description="第2个参数")

# 通过 args_schema 绑定参数模型
@tool(args_schema=FieldInfo)
def add_number(a: int, b: int) -> int:
    return a + b

# 也可自定义其他元信息
@tool(name_or_callable="my_tool", args_schema=FieldInfo, return_direct=True)
def add_number(a: int, b: int) -> int:
    return a + b
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `args_schema` | `BaseModel` | Pydantic 模型，定义工具的输入参数 schema |
| `name_or_callable` | `str` | 覆盖默认工具名（默认用函数名） |
| `return_direct` | `bool` | `True` 时工具结果直接返回给用户，不经过 LLM 再润色 |

**工具参数推导方式对比：**

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| 类型注解 `a: int` | 从 Python 类型注解自动推导 schema | 简单工具 |
| `args_schema=PydanticModel` | 用 Pydantic `Field(description=...)` 精确描述参数 | 复杂工具 / 需要详细参数说明 |


### 13.3 Pydantic StrictInt — 严格类型校验复习

> 📂 Demo：[PydanticDemo.py](./day04/08_tools/PydanticDemo.py)

**`int` vs `StrictInt` —— 工具参数类型校验的刚需：**

```python
from pydantic import BaseModel, StrictInt, ValidationError

class User(BaseModel):
    id: StrictInt   # 严格模式：拒绝类型转换
    name: str
    age: int = 0

# int: 宽松，自动转换 "41" → 41 ✅
# StrictInt: 严格，只接受真正的 int，传入字符串报错

try:
    User(id="abc", name="Bob")  # StrictInt 拒绝 "abc"
except ValidationError as e:
    print(e)
    # ❌ value is not a valid integer
```

| 类型 | 字符串 `"41"` | 布尔 `True` | 纯整数 `41` |
|------|:--:|:--:|:--:|
| `int` | ✅ 自动转换 | ✅ 转为 1 | ✅ |
| `StrictInt` | ❌ 报错 | ❌ 报错 | ✅ |

> **工具场景建议**：工具参数的 schema 中推荐用 `int`（宽松），因为 LLM 可能输出 `"41"` 而不是 `41`。仅在需要严格数据校验的业务模型中使用 `StrictInt`。


### 13.4 实战：天气查询工具

> 📂 Demo：[QueryWeatherTool.py](./day04/08_tools/QueryWeatherTool.py)

**用 @tool 封装真实 API 调用：**

```python
from langchain_core.tools import tool
import httpx
import json
import os

@tool
def get_weather(loc):
    """
    查询即时天气函数

    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称。
                注意，中国的城市需要用对应城市的英文名称代替，例如查询北京市天气，
                则 loc 参数需要输入 'Beijing'/'shanghai'。
    :return: OpenWeather API 查询即时天气的结果，JSON 格式字符串。
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn"
    }
    response = httpx.get(url, params=params, timeout=30)
    data = response.json()
    return json.dumps(data)

# 直接调用测试
result = get_weather.invoke("beijing")
```

> **关键**：docstring 中详细描述了参数含义和格式要求，LLM 会据此决定如何传参。例如用户说"北京天气"，LLM 知道该传 `"beijing"`。


### 13.5 bind_tools — 将工具绑定到 LLM

> 📂 Demo：[LLMQueryWeatherDemo.py](./day04/08_tools/LLMQueryWeatherDemo.py)

**核心流程**：`@tool` 定义工具 → `bind_tools()` 绑定到模型 → LLM 自动决定是否调用工具。

```python
# Step 1: 将工具绑定到模型
llm_with_tools = llm.bind_tools([get_weather])

# 此时 llm 调用时会自动判断：
#   - 需要查询天气 → 返回 tool_call
#   - 不需要 → 正常回复
```

| API | 说明 |
|------|------|
| `model.bind_tools([tool1, tool2, ...])` | 将工具列表绑定到模型，返回新的 Runnable |
| 返回值 | 增强了 tool-calling 能力的模型实例 |


### 13.6 JsonOutputKeyToolsParser — 解析工具调用结果

> 📂 Demo：[LLMQueryWeatherDemo.py](./day04/08_tools/LLMQueryWeatherDemo.py)

**LLM 做工具决策，Parser 提取工具调用，Tool 执行：**

```python
from langchain_core.output_parsers import JsonOutputKeyToolsParser

parser = JsonOutputKeyToolsParser(
    key_name=get_weather.name,    # 指定要提取哪个工具的输出
    first_tool_only=True           # 只取第一个匹配结果
)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `key_name` | `str` | 工具名称，解析器只提取该工具对应的 JSON |
| `first_tool_only` | `bool` | `True` 返回单个对象，`False` 返回列表 |


### 13.7 完整工具链 — 从问天气到自然语言回复

> 📂 Demo：[LLMQueryWeatherDemo.py](./day04/08_tools/LLMQueryWeatherDemo.py)

**完整流程**：用户提问 → LLM 决策 → 工具执行 → 结果格式化：

```python
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

# ① 定义工具（见 13.4）
from QueryWeatherTool import get_weather

# ② 绑定工具到模型
llm_with_tools = llm.bind_tools([get_weather])

# ③ 创建工具调用解析器
parser = JsonOutputKeyToolsParser(
    key_name=get_weather.name,
    first_tool_only=True
)

# ④ 工具调用链：模型决策 → 解析器提取 → 工具执行
get_weather_chain = llm_with_tools | parser | get_weather

# ⑤ 格式化输出链：将 JSON 结果转为自然语言
output_prompt = PromptTemplate.from_template(
    """你将收到一段 JSON 格式的天气数据{weather_json}，请用简洁自然的方式将其转述给用户。
    以下是天气 JSON 数据：
    请将其转换为中文天气描述，例如：
    "北京现在天气：多云，气温 28℃，体感有点闷热..."
    """
)
output_chain = output_prompt | llm | StrOutputParser()

# ⑥ 拼接完整链
full_chain = get_weather_chain | (lambda x: {"weather_json": x}) | output_chain

# ⑦ 执行
result = full_chain.invoke("请问北京今天的天气如何？")
# 返回自然语言天气描述
```

**链式数据流：**

```
用户输入 → llm_with_tools（判断需要查天气）
        → JsonOutputKeyToolsParser（提取 tool_call）
        → get_weather（执行 API 查询）
        → lambda 包装 → output_chain（LLM 润色为自然语言）
        → 最终回复
```

### 13.8 Tools API 速查

| API | 类型 | 说明 |
|-----|------|------|
| `@tool` | 装饰器 | 将函数注册为 LangChain 工具 |
| `@tool(args_schema=Model)` | 装饰器 | 带自定义参数 schema 的工具 |
| `tool.invoke(dict)` | 方法 | 调用工具，传入参数字典 |
| `tool.name` | 属性 | 工具名称 |
| `tool.description` | 属性 | 工具描述（取自 docstring） |
| `tool.args` | 属性 | 工具参数 schema |
| `model.bind_tools([t1, t2])` | 方法 | 绑定工具到模型 |
| `JsonOutputKeyToolsParser(key_name, first_tool_only)` | 解析器 | 从 LLM 输出中提取指定工具的调用结果 |

---

## 十四、Embedding（文本向量化）

> Embedding（嵌入）是将文本转换为数值向量的过程，是语义搜索、RAG 和相似度计算的基础。**语义相近的文本，向量距离也近**。

### 14.1 DashScope 原生调用 — 理解向量本质

> 📂 Demo：[Text2Embedding_DashScopeHello.py](./day05/09-embedding/Text2Embedding_DashScopeHello.py)

```python
import dashscope

resp = dashscope.TextEmbedding.call(
    model="text-embedding-v4",
    api_key=os.getenv("aliQwen-api"),
    input="衣服的质量杠杠的"
)

# 提取向量
embedding = resp['output']['embeddings'][0]['embedding']
# → [0.0123, -0.0045, 0.0234, ...]  高维浮点数列表
```

> **用途**：直接查看原始响应结构，理解"向量是什么样子"。

### 14.2 DashScopeEmbeddings — LangChain 统一接口（⭐ 推荐）

> 📂 Demo：[Text2Embedding_DashScope.py](./day05/09-embedding/Text2Embedding_DashScope.py)

```python
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("aliQwen-api")
)

# 单条查询向量化（用于用户问题）
query_vector = embeddings.embed_query("LangChain 怎么使用 Redis？")

# 批量文档向量化（用于构建索引）
doc_vectors = embeddings.embed_documents([
    "Redis 是一个高性能的 key-value 数据库",
    "LangChain 提供了 Redis 向量存储集成"
])
```

| 方法 | 参数 | 返回值 | 用途 |
|------|------|--------|------|
| `embed_query(text)` | `str` | `list[float]` | 将用户查询转为向量 |
| `embed_documents(texts)` | `list[str]` | `list[list[float]]` | 批量将文档块转为向量，用于建索引 |

> **关键约定**：索引阶段和查询阶段必须使用**相同的 Embedding 模型**，否则向量空间不匹配。

### 14.3 OpenAI 兼容接口 — 跨厂商切换

> 📂 Demo：[Text2Embedding_OpenAiHello.py](./day05/09-embedding/Text2Embedding_OpenAiHello.py)

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.embeddings.create(
    model="text-embedding-v3",
    input="你的文本"
)

embedding = response.data[0].embedding
```

> **融汇贯通点**：OpenAI 兼容协议在 Embedding 层同样适用——换 `base_url` 即可切换厂商。这与 Day02 的 `init_chat_model()` 和 Day06 的 MCP 协议一脉相承，都是**用统一抽象屏蔽底层差异**。

### 14.4 多模态 Embedding — 文本 + 图像

> 📂 Demo：[Text2Embedding_DashScopePro.py](./day05/09-embedding/Text2Embedding_DashScopePro.py)

```python
import dashscope

# 文本向量化
input_data = [{"text": "尚硅谷AI"}]
resp = dashscope.MultiModalEmbedding.call(
    model="multimodal-embedding-v1",
    api_key=os.getenv("aliQwen-api"),
    input=input_data,
)
embedding = resp.output["embeddings"][0]["embedding"]

# 图像向量化（需要图片 URL 或 base64）
# input_data = [{"image": "https://example.com/image.jpg"}]
```

### 14.5 余弦相似度 — 语义距离的数学计算

> 📂 Demo：[Text2Embedding_CosSimilarity.py](./day05/09-embedding/Text2Embedding_CosSimilarity.py)

**公式**：$\cos(\theta) = \frac{A \cdot B}{|A| \times |B|}$

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """余弦相似度：值越接近 1 表示语义越相似"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 实战验证
"我喜欢吃苹果" vs "苹果是我最喜欢吃的水果"  → 0.9064  # 高度相似
"我喜欢吃苹果" vs "我喜欢用苹果手机"        → 0.7656  # 中等相似
"苹果是我最喜欢吃的水果" vs "我喜欢用苹果手机" → 0.7421  # 较低相似
```

> **核心认知**：虽然三句话都包含"苹果"，但"吃苹果"和"苹果手机"的语义不同，余弦相似度精确捕捉到了这一点。这是所有语义检索的底层基础。

### 14.6 Embedding 三种调用方式对比

| 方式 | 文件 | 特点 | 适用场景 |
|------|------|------|----------|
| DashScope 原生 | `Text2Embedding_DashScopeHello.py` | 直接看原始响应，理解向量结构 | 学习底层 API |
| LangChain 封装 | `Text2Embedding_DashScope.py` | `embed_query()` / `embed_documents()` 统一接口 | 集成到 LangChain 管道 |
| OpenAI 兼容 | `Text2Embedding_OpenAiHello.py` | 换 `base_url` 即可切换厂商 | 跨厂商迁移 |

---

## 十五、向量数据库 Redis

> 向量存进去不是目的，**能快速找到最相似的**才是目的。Redis 是 LangChain 支持的向量数据库之一。

### 15.1 Redis.from_documents() — 文档流写入（⭐ 推荐）

> 📂 Demo：[EmbeddingStoreRedis.py](./day05/09-embedding/EmbeddingStoreRedis.py) | [EmbeddingRagLLM.py](./day05/10-rag/EmbeddingRagLLM.py)

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis

embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("aliQwen-api")
)

# 一步完成：向量化 + 建索引 + 写入
vector_store = Redis.from_documents(
    documents=texts,                        # List[Document]（已分割的小块）
    embedding=embeddings,
    redis_url="redis://localhost:26379",
    index_name="my_index3",                 # 索引名，写入和查询必须一致
)

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
```

| 参数 | 说明 |
|------|------|
| `documents` | 分割后的 `List[Document]`，每个 Document 包含 `page_content` 和 `metadata` |
| `embedding` | Embedding 模型实例 |
| `redis_url` | Redis 连接地址 |
| `index_name` | 索引名称，**写入和查询必须一致** |

### 15.2 add_texts() — 文本流写入

> 📂 Demo：[RedisVectorStore.py](./day05/10-rag/RedisVectorStore.py)

```python
from langchain_redis import RedisVectorStore, RedisConfig

config = RedisConfig(redis_url="redis://localhost:26379", index_name="newsgroups")
vector_store = RedisVectorStore(config, embeddings)

# 手动指定文本和元数据
texts = ["我喜欢吃苹果", "我喜欢用苹果手机"]
metadatas = [{"source": "text1"}, {"source": "text2"}]

# 写入（可先 embed_documents 预览向量）
ids = vector_store.add_texts(texts, metadatas)
# → ['newsgroups:01KKDZ5...', 'newsgroups:01KKDZ6...']
```

### 15.3 similarity_search_with_score — 带分数的语义搜索

> 📂 Demo：[RedisVectorStore_SimilaritySearch.py](./day05/10-rag/RedisVectorStore_SimilaritySearch.py)

```python
# 带分数的相似性搜索
results = vector_store.similarity_search_with_score("我喜欢用什么手机", k=3)

for doc, score in results:
    print(f"内容: {doc.page_content}")
    print(f"相似度: {1 - score:.4f}")   # 距离转相似度（仅展示用）
```

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `as_retriever(k=N)` | `Retriever` | 标准检索器，用于 LCEL 链 |
| `similarity_search(query, k=N)` | `list[Document]` | 不带分数 |
| `similarity_search_with_score(query, k=N)` | `list[(Document, float)]` | 带距离分数 |

---

## 十六、文档加载器（Document Loaders）

> 文档加载器将不同格式的文件统一加载为 `List[Document]` 对象。每个 Document 包含 `page_content`（文本正文）和 `metadata`（来源信息）。

### 16.1 通用模式

```python
loader = XxxLoader("file_path", **options)
documents = loader.load()              # → List[Document]
```

### 16.2 六种格式速查

| 格式 | 加载器 | 导入路径 | 关键参数 |
|------|--------|----------|----------|
| TXT | `TextLoader` | `langchain_community.document_loaders` | `encoding="utf-8"` |
| CSV | `CSVLoader` | `langchain_community.document_loaders.csv_loader` | `content_columns`, `metadata_columns` |
| JSON | `JSONLoader` | `langchain_community.document_loaders` | `jq_schema="."`（需 `pip install jq`） |
| DOCX | `UnstructuredWordDocumentLoader` | `langchain_community.document_loaders` | `mode="single"` / `"elements"` |
| MD | `UnstructuredMarkdownLoader` | `langchain_community.document_loaders` | `mode="elements"`（保留标题层级） |
| PDF | `PyPDFLoader` | `langchain_community.document_loaders` | `extraction_mode="plain"` / `"layout"` |

> 📂 Demo 文件：[day05/10-rag/docloads/](./day05/10-rag/docloads/) 目录下对应每种格式

### 16.3 CSV 加载最佳实践

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# 模式 1（默认）：整行作为 page_content
loader = CSVLoader("sample.csv")

# 模式 2（⭐ 推荐用于 RAG）：指定正文列和元数据列
loader = CSVLoader(
    "sample.csv",
    content_columns=["content"],          # 只有这些列进入向量化正文
    metadata_columns=["title", "author"]  # 这些列用于过滤/展示
)
```

### 16.4 PDF 加载注意事项

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf", extraction_mode="plain")
documents = loader.load()
# 每个页面一个 Document，metadata 包含 page、total_pages、source 等
```

> **注意**：PDF 是最棘手的格式。"能加载" 不等于 "适合直接用在 RAG 中"，通常需要额外的文本清洗。

---

## 十七、文本分割器（Text Splitters）

> 为什么需要分割？① 控制 Token 成本——整篇文档塞不下；② 提高检索精度——小块更容易匹配到最相关内容。

### 17.1 RecursiveCharacterTextSplitter — 核心分割器

> 📂 Demo：[RecursiveTextSplitter.py](./day05/10-rag/textsplit/RecursiveTextSplitter.py) | [RecursiveTextSplitterV2.py](./day05/10-rag/textsplit/RecursiveTextSplitterV2.py)

```python
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,        # 每块最大字符数
    chunk_overlap=30,      # 块之间的重叠字符数
    length_function=len,   # 计算长度的函数
)

# 方式 1：分割纯文本字符串 → List[str]
chunks = splitter.split_text("这是一段很长的文本内容...")

# 方式 2：分割 Document 列表 → List[Document]（⭐ 推荐）
texts = splitter.split_documents(documents)
```

### 17.2 split_text() vs split_documents()

> 📂 Demo：[RecursiveDocumentSplitter.py](./day05/10-rag/textsplit/RecursiveDocumentSplitter.py)

| 方法 | 输入 | 输出 | 元数据 |
|------|------|------|--------|
| `split_text(text)` | `str` | `list[str]` | ❌ 丢失 |
| `split_documents(docs)` | `list[Document]` | `list[Document]` | ✅ 保留 |

```python
# split_documents 保留原始 Document 的 metadata
documents = loader.load("倚天屠龙记.txt")
texts = splitter.split_documents(documents)
# texts[0].metadata → {'source': '倚天屠龙记.txt'}  # 保留！
```

> **在真实 RAG 中始终使用 `split_documents()`**，因为你需要保留来源信息用于溯源。

### 17.3 chunk_overlap 的作用

```
原始文本:  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
chunk_size=10, chunk_overlap=3

块1: "ABCDEFGHIJ"
块2: "HIJKLMNOPQR"    ← 与块1重叠 "HIJ"
块3: "OPQRSTUVWX"     ← 与块2重叠 "OPQ"
块4: "VWXYZ"
```

> **为什么需要重叠？** 避免关键信息正好落在两个块的边界上被截断。

### 17.4 CharacterTextSplitter — 简单字符分割

```python
from langchain_classic.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len
)
```

| 分割器 | 区别 | 建议 |
|--------|------|------|
| `RecursiveCharacterTextSplitter` | 按 `["\n\n", "\n", " ", ""]` 优先级递归分割 | ⭐ 通用首选 |
| `CharacterTextSplitter` | 按单一分隔符简单切割 | 快速原型 |

---

## 十八、RAG 完整管道

> RAG（Retrieval-Augmented Generation）= **检索 + 生成**。先检索相关文档，再基于文档内容生成回答。解决 LLM 的知识截止和幻觉问题。

### 18.1 完整管道代码（⭐ 核心）

> 📂 Demo：[EmbeddingRagLLM.py](./day05/10-rag/EmbeddingRagLLM.py)

```python
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import Docx2txtLoader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ① 加载文档
loader = Docx2txtLoader("alibaba-java.docx")
documents = loader.load()

# ② 分割
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# ③ 向量化 + 写入 Redis
vector_store = Redis.from_documents(
    documents=texts, embedding=embeddings,
    redis_url="redis://localhost:26379", index_name="my_index3"
)

# ④ 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ⑤ 提示词模板（{context} 由检索器填充）
prompt = PromptTemplate(
    template="""请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，
    如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。
    
    文本内容：{context}
    问题：{question}
    回答：""",
    input_variables=["context", "question"]
)

# ⑥ LCEL 链：检索 + 生成
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# ⑦ 调用
result = rag_chain.invoke("00000和A0001分别是什么意思")
print(result.content)
# → "00000 的意思是'一切 ok'，A0001 的意思是'用户端错误'"
```

**RAG 管道数据流**：

```
用户问题 → retriever(检索 top-k 文档) ──→ {context}
         → RunnablePassthrough() ──→ {question}
                                     ↓
                              prompt.format() → LLM → 回答
```

### 18.2 RAG vs 无 RAG 对比

```python
# 有 RAG（从知识库检索）
rag_chain.invoke("00000和A0001分别是什么意思")
# ✅ "00000 是'一切 ok'，A0001 是'用户端错误'"

# 无 RAG（纯靠模型自身知识）
no_rag_chain = (
    {"context": lambda _: "（未提供相关文档）", "question": RunnablePassthrough()}
    | prompt | llm
)
no_rag_chain.invoke("00000和A0001分别是什么意思")
# ❌ "抱歉，提供的文本中没有这个信息"
```

| | 有 RAG | 无 RAG |
|---|---|---|
| 知识来源 | 外部文档检索 | 模型训练数据 |
| 回答质量 | 基于文档，可溯源 | 可能编造 |
| 适用场景 | 企业知识库、私有文档问答 | 通用知识问答 |

> **RAG 的核心价值不是让模型更聪明，而是给它正确的参考资料。**

### 18.3 Day05 核心 API 速查

| 功能 | API | 来源 |
|------|-----|------|
| Embedding 查询 | `DashScopeEmbeddings.embed_query(text)` | `langchain_community` |
| Embedding 批量 | `DashScopeEmbeddings.embed_documents(texts)` | `langchain_community` |
| 余弦相似度 | `np.dot(A,B) / (np.linalg.norm(A)*np.linalg.norm(B))` | numpy |
| 文档加载 | `XxxLoader(path).load()` | `langchain_community.document_loaders` |
| 文本分割 | `RecursiveCharacterTextSplitter(chunk_size, overlap).split_documents(docs)` | `langchain_classic` |
| 写入 Redis | `Redis.from_documents(docs, embedding, url, index_name)` | `langchain_community.vectorstores` |
| 创建检索器 | `vector_store.as_retriever(search_kwargs={"k": N})` | `langchain_community.vectorstores` |
| 带分数搜索 | `vector_store.similarity_search_with_score(query, k=N)` | `langchain_community.vectorstores` |
| RAG 链 | `{"context": retriever, "question": RunnablePassthrough()} \| prompt \| llm` | LCEL |

---

## 十九、MCP 协议

> MCP（Model Context Protocol）是 Anthropic 提出的开放协议，**让 LLM 和外部工具/数据源的连接标准化**。它之于工具，就像 OpenAI 兼容接口之于模型——统一标准，消除碎片化。

### 19.1 MCP 三大原语

| 原语 | 装饰器 | 用途 | 示例 |
|------|--------|------|------|
| **Tool**（工具） | `@mcp.tool()` | LLM 可调用的函数 | 查天气、算加法 |
| **Resource**（资源） | `@mcp.resource("uri")` | 向 LLM 暴露的静态/动态数据 | 配置文件、问候语 |
| **Prompt**（提示词） | `@mcp.prompt()` | 结构化的 LLM 输入模板 | 风格化问候 |

### 19.2 FastMCP 官方框架（⭐ 推荐）

> 📂 Demo：[McpServerByFastMCP.py](./day06/11_mcp/McpServerByFastMCP.py)

```bash
pip install mcp
```

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

# 工具：LLM 可以调用的函数
@mcp.tool()
def add(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b

# 资源：向 LLM 暴露的数据
@mcp.resource("greeting://default")
def get_greeting() -> str:
    return "Hello from static resource!"

# 提示词：结构化的输入模板
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    styles = {
        "friendly": "写一句友善的问候",
        "formal": "写一句正式的问候",
        "casual": "写一句轻松的问候",
    }
    return f"为{name}{styles.get(style, styles['friendly'])}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 19.3 自定义 MCP 服务端（理解原理）

> 📂 Demo：[McpServer.py](./day06/11_mcp/McpServer.py)

```python
class MCPWeatherServer:
    def __init__(self, name, host="127.0.0.1", port=8000):
        self.name = name
        self._tools = {}

    def tool(self):
        """装饰器：注册工具函数到 _tools 字典"""
        def decorator(func):
            self._tools[func.__name__] = func
            return func
        return decorator

    def run(self, transport="sse"):
        print(f"Server running at http://{self.host}:{self.port}/sse")
        self._keep_alive()   # 无限循环等待请求

mcp = MCPWeatherServer("WeatherServer")

@mcp.tool()
def get_weather(city: str) -> str:
    """查询天气"""
    # 调用 OpenWeather API...
    return json.dumps(data)
```

| | FastMCP 官方 | 自定义实现 |
|---|---|---|
| 文件 | `McpServerByFastMCP.py` | `McpServer.py` |
| 传输 | stdio（标准输入输出） | SSE（HTTP，`http://127.0.0.1:8000/sse`） |
| 学习价值 | 生产环境开发 | 理解底层原理 |
| 依赖 | `pip install mcp` | 无外部依赖 |

### 19.4 MCP 客户端调用

> 📂 Demo：[McpClient.py](./day06/11_mcp/McpClient.py)

```python
class MCPWeatherClient:
    def __init__(self, mcp_instance):
        self._tools = mcp_instance._tools

    def check_tool_availability(self, tool_name):
        return tool_name in self._tools

    def call_get_weather(self, city):
        tool_func = self._tools["get_weather"]
        return tool_func(city)

# 遍历多个城市
for city in ["Beijing", "Shanghai"]:
    weather = client.call_get_weather(city)
```

> **融汇贯通点**：MCP 之于工具 = OpenAI 兼容接口之于模型。将来源头不同的工具（天气 API、数据库、文件系统），通过统一的 MCP Server 暴露，任何支持 MCP 的 LLM 客户端都能直接消费。

---

## 二十、Agent（智能体）

> Agent 让 LLM 从"被动回答"升级为"自主行动"——它会**推理**需要什么、**选择**工具、**观察**结果、**决策**下一步，循环直到完成任务。

### 20.1 create_agent() — LangChain 1.0 Agent 创建（⭐ 推荐）

> 📂 Demo：[AgentReact.py](./day06/12_agent/AgentReact.py) | [AgentSmartSelectV1.0.py](./day06/12_agent/AgentSmartSelectV1.0.py)

```python
from langchain.agents import create_agent
from langchain.tools import tool

# 定义工具
@tool
def search_products(query: str) -> str:
    """搜索产品并返回按受欢迎度排序的结果"""
    # ... 业务逻辑
    return formatted_result

@tool
def check_inventory(product_id: str) -> str:
    """检查特定产品的库存状态"""
    # ... 业务逻辑
    return stock_info

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[search_products, check_inventory],
    system_prompt="你是电商助手，遵循ReAct模式：推理 → 行动 → 观察 → 重复"
)

# 调用
result = agent.invoke({
    "messages": [{"role": "user", "content": "查找最受欢迎的无线耳机并检查库存"}]
})
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `ChatModel` | 驱动 Agent 的语言模型 |
| `tools` | `list` | 可用工具列表（`@tool` 装饰的函数） |
| `system_prompt` | `str` | 系统提示词，定义 Agent 的角色和行为规则 |
| `response_format` | `BaseModel` / `TypedDict` | 可选，结构化输出模式（见 21.2） |

### 20.2 ReAct 模式 — Agent 的思维循环

```
用户: "查找最受欢迎的无线耳机并检查是否有库存"
  │
  ▼  🧠 推理(Reasoning)
  │   "用户要找无线耳机，需要先搜索产品"
  ▼  🛠️ 行动(Acting)
  │   search_products("无线耳机") → 返回5个产品
  ▼  👁️ 观察(Observation)
  │   "最受欢迎：索尼 WH-1000XM5, ID=WH-1000XM5, 95%"
  ▼  🧠 推理
  │   "这个产品最受欢迎，现在检查库存"
  ▼  🛠️ 行动
  │   check_inventory("WH-1000XM5") → 有库存，10件
  ▼  ✅ 最终回答
      "最受欢迎的是索尼 WH-1000XM5，售价¥299，库存10件"
```

**追踪 ReAct 循环的消息类型**：

```python
for msg in result['messages']:
    if msg.type == "AIMessage" and msg.tool_calls:    # 推理+行动
        print(f"🛠️ 工具调用: {msg.tool_calls}")
    elif msg.type == "ToolMessage":                    # 观察
        print(f"📋 观察结果: {msg.content}")
    elif msg.type == "AIMessage" and not msg.tool_calls:  # 最终回答
        print(f"✅ 回答: {msg.content}")
```

### 20.3 经典 Agent API（V0.3 风格）

> 📂 Demo：[AgentSmartSelectV0.3.py](./day06/12_agent/AgentSmartSelectV0.3.py)

```python
from langchain_classic import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是天气助手..."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke({"input": "请问北京和上海今天天气怎么样？"})
```

| API | 版本 | 特点 |
|------|------|------|
| `create_agent()` | LangChain 1.0（⭐ 推荐） | 简洁、支持 `response_format` |
| `create_tool_calling_agent()` + `AgentExecutor` | langchain_classic | 兼容旧版本，verbose 日志 |

### 20.4 并行工具调用

Agent 可以在**单次推理**中判断需要调用多个工具（甚至同一个工具多次），并行触发：

```python
# 用户: "北京和上海今天天气怎么样，哪个城市更热？"
# Agent 自动并行调用：
#   get_weather("Beijing")  ──┐
#   get_weather("Shanghai") ──┤ 同时执行
#                              ↓
#                  比较温度 → 回答
```

---

## 二十一、Agent 高级模式

### 21.1 A2A 多智能体协作

> 📂 Demo：[Agent2Agent.py](./day06/12_agent/Agent2Agent.py)

**模式**：A2A 调度 = **多个功能单一的 Runnable 子 Agent 链 + 一个控制调用逻辑的总协调器**。

```
用户: "安排北京飞上海的完整行程"
  │
  ▼
┌──────────────────────┐
│   总协调 Agent        │  ← RunnableLambda，顺序编排
└──────────────────────┘
  │         │         │
  ▼         ▼         ▼
┌──────┐ ┌──────┐ ┌──────┐
│携程   │ │美团   │ │滴滴   │  ← 子 Agent，单一职责，单工具绑定
│机票   │ │酒店   │ │打车   │
└──────┘ └──────┘ └──────┘
```

**子 Agent 模板**：

```python
def create_ctrip_agent(llm):
    """子 Agent = prompt | llm_with_tools | output_parser"""
    llm_with_tools = llm.bind_tools([ctrip_book_flight])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业的工具调用助手，只能调用CtripBookFlight工具..."),
        ("human", "{input}")
    ])
    return prompt | llm_with_tools | StrOutputParser()
```

**总协调器核心逻辑**：

```python
def create_travel_coordinator_agent(llm, ctrip_chain, meituan_chain, didi_chain):
    def a2a_schedule(input_dict):
        # 1. 携程
        ctrip_result = ctrip_chain.invoke({"input": "订机票"})
        # 2. 美团
        meituan_result = meituan_chain.invoke({"input": "订酒店"})
        # 3. 滴滴
        didi_result = didi_chain.invoke({"input": "预约打车"})
        # 整合报告
        return f"【最终报告】\n{ctrip_result}\n{meituan_result}\n{didi_result}"

    return RunnableLambda(a2a_schedule)
```

**A2A 四大核心原则**：

| 原则 | 说明 |
|------|------|
| **单一职责** | 一个子 Agent 只负责一个业务，只绑定一个专属工具 |
| **统一接口** | 所有子 Agent 封装为 `prompt \| llm_with_tools \| output_parser`，对外仅暴露 `invoke()` |
| **集中调度** | 总协调器控制所有调用顺序，子 Agent 之间不直接交互 |
| **空值兜底** | 每个调用加 try-except，空结果通过 `tool.func` 获取原始函数兜底 |

**兜底机制**：

```python
# 通过 .func 获取 @tool 装饰器下的原始函数
ctrip_func = ctrip_book_flight.func

# Agent 返回空时，直接调用原始函数
try:
    result = ctrip_chain.invoke({"input": "订机票"})
except:
    result = ""
if not result.strip():
    result = ctrip_func("北京", "上海", "2026-02-01")  # 兜底
```

### 21.2 结构化输出 Agent

> 📂 Demo：[AgentSmartSelectV1.0.py](./day06/12_agent/AgentSmartSelectV1.0.py)

**V0.3 → V1.0 关键升级**：Agent 输出从自然语言变为 TypedDict 结构化对象，下游代码可直接消费。

```python
from typing import TypedDict

# 定义输出结构
class WeatherCompareOutput(TypedDict):
    beijing_temp: float
    shanghai_temp: float
    hotter_city: str
    summary: str

# 创建 Agent，指定结构化输出
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是天气助手...",
    response_format=WeatherCompareOutput,   # ← 关键参数
)

result = agent.invoke({"input": "请问北京和上海今天天气怎么样，哪个更热？"})

# 直接获取结构化数据
structured = result["structured_response"]
# → {'beijing_temp': 32.0, 'shanghai_temp': 35.0, 'hotter_city': '上海', 'summary': '...'}

print(json.dumps(structured, ensure_ascii=False, indent=2))
```

| | V0.3（经典） | V1.0（结构化） |
|---|---|---|
| API | `create_tool_calling_agent` + `AgentExecutor` | `create_agent()` |
| 输出格式 | 自然语言文本 | TypedDict 结构化对象 |
| 可编程性 | 需要解析自然语言 | 直接 `result["structured_response"]` |
| 适用场景 | 人类阅读 | 下游代码消费 |

### 21.3 Agent vs RAG 的融合视角

```
用户请求
  │
  ▼
Agent 推理（Day06）──→ 需要查资料？──→ RAG 检索知识库（Day05）
  │                                     │
  │                    ┌────────────────┘
  │                    ▼
  ├──→ 需要调 API？──→ MCP 工具（Day06）──→ 天气/机票/酒店...
  │
  ▼
综合所有信息 → 决策 → 回答
```

**一句话总结**：Day05 让你**喂对资料**，Day06 让你**做对决策**——两者结合，就是现代 AI Agent 应用的完整技术栈。

### 21.4 Day06 核心 API 速查

| 功能 | API | 来源 |
|------|-----|------|
| FastMCP 工具 | `@mcp.tool()` | `mcp.server.fastmcp` |
| FastMCP 资源 | `@mcp.resource("uri")` | `mcp.server.fastmcp` |
| FastMCP 提示词 | `@mcp.prompt()` | `mcp.server.fastmcp` |
| 创建 Agent（1.0） | `create_agent(model, tools, system_prompt, response_format)` | `langchain.agents` |
| 创建 Agent（经典） | `create_tool_calling_agent(llm, tools, prompt)` + `AgentExecutor` | `langchain_classic` |
| 工具绑定 | `llm.bind_tools([tool1, tool2])` | `langchain_openai` |
| 子 Agent 链 | `prompt \| llm_with_tools \| StrOutputParser` | LCEL |
| A2A 协调器 | `RunnableLambda(a2a_schedule)` | `langchain_core.runnables` |
| 工具原始函数 | `tool_object.func` | `@tool` 装饰器内部 |
| 结构化输出 | `create_agent(..., response_format=TypedDict)` | `langchain.agents` |

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

# ==================== LCEL（Day04） ====================
from langchain_core.runnables import (
    RunnableParallel,          # 并行链
    RunnableLambda,            # 函数链
    RunnableBranch,            # 分支链
    RunnableConfig,            # 运行时配置
)
from langchain_core.runnables.history import RunnableWithMessageHistory  # 带记忆链

# ==================== 对话记忆（Day04） ====================
from langchain_core.chat_history import InMemoryChatMessageHistory      # 内存会话历史
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory,   # Redis 持久化会话历史
)

# ==================== 工具定义（Day04） ====================
from langchain_core.tools import tool                                   # @tool 装饰器
from langchain_core.output_parsers import JsonOutputKeyToolsParser      # 工具调用解析器
# model.bind_tools([tool1, tool2])   将工具绑定到模型
# tool.invoke({"arg": value})        调用工具

# ==================== Embedding（Day05） ====================
from langchain_community.embeddings import DashScopeEmbeddings           # 阿里云 DashScope 嵌入
import dashscope                                                         # DashScope 原生 SDK
from dashscope import TextEmbedding, MultiModalEmbedding                 # 文本/多模态嵌入

# ==================== 向量数据库（Day05） ====================
from langchain_community.vectorstores import Redis                       # Redis 向量存储
from langchain_redis import RedisVectorStore, RedisConfig               # langchain_redis 封装

# ==================== 文档加载器（Day05） ====================
from langchain_community.document_loaders import (
    TextLoader,                          # TXT
    JSONLoader,                          # JSON
    UnstructuredWordDocumentLoader,      # DOCX
    UnstructuredMarkdownLoader,          # Markdown
    PyPDFLoader,                         # PDF
)
from langchain_community.document_loaders.csv_loader import CSVLoader   # CSV

# ==================== 文本分割器（Day05） ====================
from langchain_classic.text_splitter import (
    RecursiveCharacterTextSplitter,      # 递归分割（⭐ 推荐）
    CharacterTextSplitter,               # 简单字符分割
)

# ==================== RAG（Day05） ====================
from langchain_core.runnables import RunnablePassthrough                # 直通节点
# rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

# ==================== MCP 协议（Day06） ====================
from mcp.server.fastmcp import FastMCP                                   # FastMCP 框架
# @mcp.tool()      注册工具
# @mcp.resource()  注册资源
# @mcp.prompt()    注册提示词模板

# ==================== Agent（Day06） ====================
from langchain.agents import create_agent                                # LangChain 1.0 Agent（⭐ 推荐）
from langchain_classic import create_tool_calling_agent                  # 经典 Agent 创建
from langchain_classic.agents import AgentExecutor                       # Agent 执行器

# ==================== 调用方式 ====================
# model.invoke(input)          同步单次
# model.ainvoke(input)         异步单次
# model.batch(inputs)          同步批量
# model.abatch(inputs)         异步批量
# model.stream(input)          同步流式
# model.astream(input)         异步流式
# model.with_structured_output(Schema)   结构化输出
```

> 📅 文档更新时间：2026-07-30 — 基于尚硅谷 LangChain Day01~Day06 课程内容整理
> 
> Day01-04（初始版本，2026-07-22）→ Day05（Embedding + 向量数据库 + RAG）→ Day06（MCP 协议 + Agent）