from langgraph.graph import StateGraph, START, END

def add(state):
    print("*"*80)
    print("before add:"+str(state))
    state["y"] = state["x"]+1
    print("after add:" + str(state))
    return state


def buildGraph():
    #定义图的时候，要指定状态的数据类型
    graphBuilder = StateGraph(dict)

    graphBuilder.add_node("add", add)

    graphBuilder.add_edge(START,"add")
    graphBuilder.add_edge("add", END)

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    from courseCode import showGraph

    showGraph.showGraphInCode(graph, "state1.jpg")

    #state本质上是一个字典，可以使用python的原生字典
    state = {"x":2}
    result = graph.invoke(state)
    print(state)