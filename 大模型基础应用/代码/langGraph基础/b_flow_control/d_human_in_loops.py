from langgraph.graph import START,StateGraph,  END
from typing_extensions import TypedDict
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver

#添加状态类
class State(TypedDict):
    产物:str
    是否加糖:str

#冲咖啡函数
def 冲咖啡(state):
    print("冲咖啡之前:" + str(state))
    state["产物"]="咖啡"
    print("冲咖啡之后:" + str(state))
    return state

#中断节点，里面必须包含一个interrupt
def 询问是否加糖(state):
    human_response = interrupt("")
    state["是否加糖"] = human_response
    return state

#根据人类是否加糖的反馈，选择不同分支
def 是否加糖分支(state):
    #print(state)
    if state["是否加糖"] == "是":
        return "是"
    elif state["是否加糖"] == "否":
        return "否"

#加糖函数
def 加糖(state):
    print("加糖之前:" + str(state))
    state["产物"]="加糖咖啡"
    print("加糖之后:" + str(state))
    return state


def buildGraph():
    #初始化图
    graphBuilder = StateGraph(State)
    #初始化记忆。使用人工介入必须带有记忆，否则图中断执行后无法正常继续
    checkpointer = InMemorySaver()
    #添加节点
    graphBuilder.add_node("冲咖啡", 冲咖啡)
    graphBuilder.add_node("询问是否加糖", 询问是否加糖)
    graphBuilder.add_node("加糖", 加糖)
    #添加边
    graphBuilder.add_edge(START, "冲咖啡")
    graphBuilder.add_edge("冲咖啡", "询问是否加糖")
    graphBuilder.add_conditional_edges("询问是否加糖", 是否加糖分支,{"是":"加糖","否":END})
    graphBuilder.add_edge("加糖", END)
    #编译图
    graph = graphBuilder.compile(checkpointer=checkpointer)
    return graph

if __name__ == "__main__":
    graph = buildGraph()
    #打印图
    import showGraph

    showGraph.showGraphInCode(graph, "human_in_loop.jpg")
    #调用图
    state:State = {"产物":"开水","是否加糖":""}
    config = {"configurable": {"thread_id": "some_id"}}

    #启动工作流
    state = graph.invoke(state, config)
    # 中断后，接收用户输入
    userDecision = input("先生您好，请问您的咖啡需要加糖吗？(是/否):")
    while userDecision not in {"是", "否"}:
        userDecision = input("先生您好，请问您的咖啡需要加糖吗？(是/否):")
    #使用Command函数向graph提供用户输入的数据，继续工作流
    result = graph.invoke(Command(resume=userDecision), config=config)
    print(result)