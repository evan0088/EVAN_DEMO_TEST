
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationChain

from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType



def useLLM():
    #使用大模型
    model = OllamaLLM(model="qwen2.5:7b", temperature=0)
    result = model.invoke("请给我讲个鬼故事")
    print(result)

def useChatLLM():
    #使用聊天模型
    model = ChatOllama(model="qwen2.5:7b", temperature=0)
    messages = [
            SystemMessage(content="现在你是一个著名的歌手"),
            HumanMessage(content="给我写一首歌词")
    ]
    res = model.invoke(messages)
    print(res)
    print(res.content)

def useEmbeddingModel():
    #使用嵌入模型
    model = OllamaEmbeddings(model="bge-m3", temperature=0)
    res1 = model.embed_query('这是第一个测试文档')
    print(res1)

    res2 = model.embed_documents(['这是第一个测试文档', '这是第二个测试文档'])
    print(res2)


def usePromptTemplateZeroShot():
    # 定义模板
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

    prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template,
    )
    prompt_text = prompt.format(lastname="王")

    print(prompt_text)
    # result: 我的邻居姓王，他生了个儿子，给他儿子起个名字

    model = OllamaLLM(model="qwen2.5:7b", temperature=0)
    result = model.invoke(prompt_text)
    print(result)

def usePromptTemplateFewShot():
    examples = [
        {"word": "开心", "antonym": "难过"},
        {"word": "高", "antonym": "矮"},
    ]

    example_template = """
    单词: {word}
    反义词: {antonym}\\n
    """

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_template,
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="给出每个单词的反义词",
        suffix="单词: {input}\\n反义词:",
        input_variables=["input"],
        example_separator="\\n",
    )

    prompt_text = few_shot_prompt.format(input="粗")
    print(prompt_text)
    print('*' * 80)
    model = OllamaLLM(model="qwen2.5:7b", temperature=0)
    print(model.invoke(prompt_text))

def useChatPromptTemplate():
    #使用提示词模板

    # 创建原始模板
    template_str = """您是一位专业的鲜花店文案撰写员。\n
    对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
    # """

    # 根据原始模板创建LangChain提示模板
    promp_emplate = ChatPromptTemplate.from_template(template_str)
    prompt = promp_emplate.format_messages(price='50', flower_name=["玫瑰"], )
    print('prompt-->', prompt)

    # 实例化模型
    model = ChatOllama(model="qwen2.5:7b", temperature=0)

    # 打印结果
    result = model.invoke(prompt)
    print(result.content)

def useMemory():
    history = ChatMessageHistory()
    history.add_user_message("在吗？")
    history.add_ai_message("有什么事?")
    print(history.messages)

def convWithMemory():
    #  实例化大模型
    llm = OllamaLLM(model="qwen2.5:7b")
    conversation = ConversationChain(llm=llm)
    resut1 = conversation.predict(input="小明有1只猫")
    print(resut1)
    print('*' * 80)
    resut2 = conversation.predict(input="小刚有2只狗")
    print(resut2)
    print('*' * 80)
    resut3 = conversation.predict(input="小明和小刚一共有几只宠物?")
    print(resut3)
    print('*' * 80)

def useFileLoader():
    loader = UnstructuredLoader('衣服属性.txt', encoding='utf8')
    docs = loader.load()
    print(docs)
    print(len(docs))
    first_01 = docs[0].page_content[:4]
    print(first_01)
    print('*' * 80)
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader('衣服属性.txt', encoding='utf8')
    docs = loader.load()
    print(docs)
    print(len(docs))
    first_01 = docs[0].page_content[:4]
    print(first_01)

def useSplitter():
    text_splitter = CharacterTextSplitter(
        separator=" ",  # 空格分割，但是空格也属于字符
        chunk_size=5,
        chunk_overlap=0,
    )

    # 一句分割
    a = text_splitter.split_text("a b c d e f")
    print(a)
    # ['a b c', 'd e f']

    # 多句话分割（文档分割）
    texts = text_splitter.create_documents(["a b c d e f", "e f g h"], )
    print(texts)

def useVectorStore():
    with open('./pku.txt',encoding="utf-8") as f:
        state_of_the_union = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)
    print(texts)

    embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)

    docsearch = Chroma.from_texts(texts, embeddings)

    query = "1937年北京大学发生了什么？"
    docs = docsearch.similarity_search(query)
    print(docs)

def useChain():
    # 定义模板
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

    prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template,
    )
    llm = OllamaLLM(model="qwen2.5:7b")

    chain = prompt|llm
    # 执行链
    print(chain.invoke("王"))

def useAgent():
    #  实例化大模型
    llm = OllamaLLM(model="qwen2.5:7b")

    #  设置工具
    # "serpapi"实时联网搜素工具、"math": 数学计算的工具
    # tools = load_tools(["serpapi", "llm-math"], llm=llm)
    tools = load_tools(["llm-math"], llm=llm)

    # 实例化代理Agent:返回 AgentExecutor 类型的实例
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print('agent', agent)
    # 准备提示词
    from langchain_core.prompts import PromptTemplate
    prompt_template = """解以下方程：3x + 4(x + 2) - 84 = y; 其中x为3，请问y是多少？"""
    prompt = PromptTemplate.from_template(prompt_template)
    print('prompt-->', prompt)

    # 代理Agent工作
    result = agent.invoke(prompt)
    print(result)

if __name__ == '__main__':
    #useLLM()
    #useChatLLM()
    #useEmbeddingModel()
    #usePromptTemplateZeroShot()
    #usePromptTemplateFewShot()
    #useChatPromptTemplate()
    #useMemory()
    #convWithMemory()

    #useFileLoader()
    #useSplitter()
    #useVectorStore()

    #useChain()
    useAgent()