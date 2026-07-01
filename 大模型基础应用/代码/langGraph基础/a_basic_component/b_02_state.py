from langgraph.graph import StateGraph,START,  END
from typing_extensions import TypedDict

#定义一个类，类型是TypedDict。这种数据类型可以方便数据校验，以后我们都采用这种方式定义状态
class State(TypedDict):
    x:int
    y:int

def add(state):
    print("*"*80)
    print("before add:"+str(state))
    state["y"] = state["x"]+1
    print("after add:" + str(state))
    return state

def buildGraph():
    # 定义图的时候，要指定状态的数据类型
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("add", add)

    graphBuilder.add_edge(START,"add")
    graphBuilder.add_edge("add", END)

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    from courseCode import showGraph

    showGraph.showGraphInCode(graph, "state2.jpg")

    #实例化状态
    state:State = {'x':2,'y':0}
    result = graph.invoke(state)
    print(result)