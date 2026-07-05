from langgraph.graph import START,StateGraph,  END
from typing_extensions import TypedDict

import showGraph


#添加状态类
class State(TypedDict):
    水温: int
    产物:str

#烧水函数
def 烧水(state):
    print("*"*80)
    print("烧水之前:"+str(state))
    state["水温"] = state["水温"]+10
    if state["水温"]>100:
        state["水温"] = 100
        state["产物"] = "烧开的水"
    print("烧水之后:" + str(state))
    return state

#按照温度处理水的条件函数
def 按温度处理水(state):
    print("before processWater:"+str(state))
    if state["水温"]== 100:
        return "水烧开了"
    else:
        return "水没开"

#冲咖啡函数
def 冲咖啡(state):
    print("冲咖啡之前:" + str(state))
    state["产物"] = "咖啡"
    print("冲咖啡之后:"+str(state))
    return state

#需要继续加热函数
def 需要继续加热(state):
    state["产物"] = "没烧开的水"
    print("继续加热"+str(state))
    return state


#构建图
def buildGraph():
    #初始化图
    graphBuilder = StateGraph(State)
    #添加节点
    graphBuilder.add_node("烧水", 烧水)
    graphBuilder.add_node("冲咖啡", 冲咖啡)
    # graphBuilder.add_node("需要继续加热", 需要继续加热)
    #添加边
    graphBuilder.add_edge(START, "烧水")
    graphBuilder.add_conditional_edges("烧水",按温度处理水,{"水烧开了":END,"水没开":"烧水"})
    # graphBuilder.add_edge("冲咖啡",END)
    # graphBuilder.add_edge("需要继续加热", END)
    #编译图
    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    #打印图

    showGraph.showGraphInCode(graph, "branch.jpg")

    #调用图
    state:State = {"水温": 28,"产物":"没烧开的水"}
    result = graph.invoke(state)
    print(result)