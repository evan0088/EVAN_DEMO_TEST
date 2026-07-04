from langchain.agents import create_agent
from langgraph.graph import StateGraph,START,  END
from typing_extensions import TypedDict

class State(TypedDict):
    x: int
    query:str
    answer:str

#函数节点本质上就是一个通用的python函数
def add(state):
    print("*"*80)
    print("before add:"+str(state))
    state["y"] = state["x"]+1
    print("after add:" + str(state))

#工具节点的作用是封装外部工具或API
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

#大模型节点调用大模型生成结果，如果需要使用工具，可以调用时加以说明
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0.6)

def configLLM(state):
    agent = create_agent(
        model=llm,
        #这里说明可以使用的工具
        tools=[get_weather],
        system_prompt="You are a helpful assistant"
    )
    #调用大模型
    result = agent.invoke(
        {"messages": [{"role": "user", "content": state["query"]}]}
    )
    state["answer"] = result["messages"][-1].content
    return state

def buildGraph():
    graphBuilder = StateGraph(State)

    #在图里添加一个节点
    graphBuilder.add_node("add", add)

    graphBuilder.add_edge(START,"add")
    graphBuilder.add_edge("add", END)

    graph = graphBuilder.compile()
    return graph
def buildGraph2():
    graphBuilder = StateGraph(State)

    # 在图里添加一个节点
    graphBuilder.add_node("configLLM", configLLM)

    graphBuilder.add_edge(START, "configLLM")
    graphBuilder.add_edge("configLLM", END)

    graph = graphBuilder.compile()
    return graph

def testGraph1():
    graph = buildGraph()

    from courseCode import showGraph
    showGraph.showGraphInCode(graph, "node1.jpg")

    # 实例化状态
    state: State = {'x': 2}
    result = graph.invoke(state)
    print(state)

def testGraph2():
    graph = buildGraph2()
    from courseCode import showGraph
    showGraph.showGraphInCode(graph, "node2.jpg")

    # 实例化状态
    state: State = {'query': 'what is the weather in Beijing'}
    result = graph.invoke(state)
    print(result)

if __name__ == "__main__":
    #testGraph1()
    testGraph2()