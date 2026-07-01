from langgraph.graph import START,StateGraph,  END
from typing_extensions import TypedDict


class State(TypedDict):
    tempreture: int

def heatUp(state):
    print("*"*80)
    print("before heatUp:"+str(state))
    state["tempreture"] = state["tempreture"]+10
    if state["tempreture"]>100:
        state["tempreture"] = 100
    print("after heatUp:" + str(state))
    return state

#条件节点的第一种方式，每个分支返回一个字符串
def processWater(state):
    if state["tempreture"]== 100:
        return "hotWaterReady"
    else:
        return "stillCode"

#条件节点的第二种方式，每个分支返回后续节点名
def processWater2(state):
    if state["tempreture"]== 100:
        return END
    else:
        return "heatUp"

def getBolidWater(state):
    return state

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("heatUp", heatUp)
    graphBuilder.add_node("getBolidWater", getBolidWater)

    # 用这种方式设置普通节点间的联系
    # graphBuilder.add_edge("add", "multiply")

    # 可以用这两种方式设置起始节点，个人建议采用第一种，符合边的通用定义方式
    graphBuilder.add_edge(START, "heatUp")
    #graphBuilder.set_entry_point("add")

    #条件边和条件节点有两种写法，个人建议第一种。所有节点流转全部集中在图的定义部分，方便检查
    #条件边的第一种方式：（起点，终点，映射表），注意：终点是条件节点，不用注册。映射表是条件节点返回的字符串映射到后续节点
    graphBuilder.add_conditional_edges("heatUp",processWater,{"hotWaterReady":"getBolidWater","stillCode":"heatUp"})

    # 条件边的第二种方式：（起点，终点），注意：终点是条件节点，不用注册。
    #graphBuilder.add_conditional_edges("heatUp", processWater)

    # 可以用这两种方式设置终止节点，个人建议采用第一种，符合边的通用定义方式
    graphBuilder.add_edge("getBolidWater",END)
    #graphBuilder.set_finish_point("getBolidWater")

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    from courseCode import showGraph

    showGraph.showGraphInCode(graph, 'edge1.png')

    state:State = {"tempreture": 28}
    result = graph.invoke(state)
    print(result)