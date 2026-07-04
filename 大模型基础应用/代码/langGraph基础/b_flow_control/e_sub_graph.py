from langgraph.graph import START,StateGraph,  END
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import InMemorySaver

import showGraph


#合并温度值的方式，取其大者
def updateTempreture(left, right):
    return max(left, right)

#合并产物的方式，优先取工序靠后的产物
def updateProduct(left,right):
    if left=="加糖咖啡" or right == "加糖咖啡":
        return "加糖咖啡"
    elif left=="咖啡" or right == "咖啡":
        return "咖啡"
    elif left=="开水" or right == "开水":
        return "开水"
    elif left=="温水" or right == "温水":
        return "温水"
    else:
        return "凉水"
    if right :
        return right
    else:
        return left

#合并咖啡固体的方式，优先取工序靠后的产物
def updateSolid(left,right):
    if left=="咖啡粉" or right == "咖啡粉":
        return "咖啡粉"
    else:
        return "咖啡豆"
def updateSugur(left,right):
    if left=="是" or right == "是":
        return "是"
    else:
        return "否"

#添加状态类
class State(TypedDict):
    水温: Annotated[int,updateTempreture]
    产物:Annotated[str,updateProduct]
    咖啡固体:Annotated[str,updateSolid]
    是否加糖:Annotated[str,updateSugur]

#磨咖啡豆函数
def 磨咖啡豆(state):
    print("*" * 80)
    print("磨咖啡豆之前:" + str(state))
    state["咖啡固体"] = "咖啡粉"
    print("磨咖啡豆之后:" + str(state))
    return state

#烧水函数
def 烧水(state):
    print("*"*80)
    print("烧水之前:"+str(state))
    state["产物"] = "没烧开的水"
    state["水温"] = state["水温"]+10
    if state["水温"]>100:
        state["水温"] = 100
        state["产物"] = "开水"
    print("烧水之后:" + str(state))
    return state

#按照温度处理水的条件函数
def 按温度处理水(state):
    if state["水温"]== 100:
        return "水烧开了"
    else:
        return "水没开"

#得到开水函数，注意这个函数没有实际作用，只是便于展示整个流程
def 得到开水(state):
    return state

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

def buildGraph5():
    # 初始化图
    graphBuilder = StateGraph(State)
    checkpointer = InMemorySaver()

    graphBuilder.add_node("heat water", 烧水)
    graphBuilder.add_node("get boil water", 得到开水)
    graphBuilder.add_node("磨咖啡豆", 磨咖啡豆)
    graphBuilder.add_node("冲咖啡", 冲咖啡)
    graphBuilder.add_node("询问是否加糖", 询问是否加糖)
    graphBuilder.add_node("加糖", 加糖)

    graphBuilder.add_edge(START, "磨咖啡豆")
    graphBuilder.add_edge(START, "heat water")
    graphBuilder.add_conditional_edges("heat water", 按温度处理水, {"水烧开了": "get boil water", "水没开": "heat water"})
    graphBuilder.add_edge(["get boil water","磨咖啡豆"], "冲咖啡")
    graphBuilder.add_edge("冲咖啡", "询问是否加糖")
    graphBuilder.add_conditional_edges("询问是否加糖", 是否加糖分支,{"是":"加糖","否":END})
    graphBuilder.add_edge("加糖", END)

    graph = graphBuilder.compile(checkpointer=checkpointer)
    return graph

def buildGraph6():
    #烧水子图
    # 初始化图
    heatWaterSubGraphBuilder = StateGraph(State)
    # 添加节点
    heatWaterSubGraphBuilder.add_node("heat water", 烧水)
    heatWaterSubGraphBuilder.add_node("get boil water", 得到开水)
    # 添加边
    heatWaterSubGraphBuilder.add_edge(START, "heat water")
    heatWaterSubGraphBuilder.add_conditional_edges("heat water", 按温度处理水,{"水烧开了": "get boil water", "水没开": "heat water"})
    # 编译图
    heatWaterSubGraph = heatWaterSubGraphBuilder.compile()

    #加糖子图
    # 初始化图
    addSugurSubGraphBuilder = StateGraph(State)
    # 初始化记忆。使用人工介入必须带有记忆，否则图中断执行后无法正常继续
    checkpointer = InMemorySaver()
    # 添加节点
    addSugurSubGraphBuilder.add_node("询问是否加糖1", 询问是否加糖)
    addSugurSubGraphBuilder.add_node("加糖", 加糖)
    # 添加边
    addSugurSubGraphBuilder.add_edge(START,"询问是否加糖1")
    addSugurSubGraphBuilder.add_conditional_edges("询问是否加糖1", 是否加糖分支, {"是": "加糖", "否": END})
    addSugurSubGraphBuilder.add_edge("加糖", END)
    # 编译图，需要带有记忆
    addSugurSubGraph = addSugurSubGraphBuilder.compile()

    #总图
    # 初始化图
    graphBuilder = StateGraph(State)
    # 添加节点
    #注意这里把子图加入总图的方式，是把编译好的子图作为一个节点加入进来
    graphBuilder.add_node("得到热水子图",heatWaterSubGraph)
    graphBuilder.add_node("磨咖啡豆", 磨咖啡豆)
    graphBuilder.add_node("冲咖啡", 冲咖啡)
    graphBuilder.add_node("加糖子图1", addSugurSubGraph)
    # 添加边
    graphBuilder.add_edge(START,"得到热水子图")
    graphBuilder.add_edge(START, "磨咖啡豆")
    graphBuilder.add_edge(["得到热水子图","磨咖啡豆"],"冲咖啡")
    graphBuilder.add_edge("冲咖啡","加糖子图1")
    # 编译图,需要带有记忆
    graph = graphBuilder.compile(checkpointer=checkpointer)
    return heatWaterSubGraph,addSugurSubGraph,graph


if __name__ == "__main__":
    """graph = buildGraph5()
    #打印图
    showGraph.showGraphInCode(graph, "复杂流程.jpg")"""

    heatWaterSubGraph,addSugurSubGraph,graph = buildGraph6()
    #打印图
    showGraph.showGraphInCode(heatWaterSubGraph,"烧热水子图.jpg")
    showGraph.showGraphInCode(addSugurSubGraph,"加糖子图.jpg")
    showGraph.showGraphInCode(graph,"总图.jpg")

    #初始化状态
    state:State = {"水温": 58,"产物":"凉水","咖啡固体":"咖啡豆"}
    #添加config
    config = {"configurable": {"thread_id": "some_id"}}
    #调用图
    state = graph.invoke(state, config)
    #中断后，接收用户输入
    userDecision = input("先生您好，请问您的咖啡需要加糖吗？(是/否):")
    while userDecision not in {"是", "否"}:
        userDecision = input("先生您好，请问您的咖啡需要加糖吗？(是/否):")
    #使用Command函数向graph提供用户输入的数据，继续工作流
    result = graph.invoke(Command(resume=userDecision), config=config)
    print(result)