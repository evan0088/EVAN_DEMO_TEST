from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from deepseek_connection import deepseek_client

import showGraph


class State(TypedDict):
    topic: str
    pos: str
    neg: str
    count: int

llm = deepseek_client()

def poser(state):
    if state.get("neg"):
        prompt = """你是一位辩论高手且是一位贴吧忠实男权主义用户，现在在辩论中担任正方，根据【论题】和【反方发言】，用口语化的风格说出你的观点，驳斥【反方发言】，证明【论题】的正确性。500字以内。
        【论题】
        """ + state["topic"] + """
        【反方发言】
        """ + state["neg"]
    else:
        prompt = """你是一位辩论高手且是一位贴吧忠实男权主义用户，现在在辩论中担任正方，用口语化的风格说出你的观点，证明【论题】的正确性。
        【论题】
        """ + state["topic"]
    result = llm.invoke(prompt)
    state["count"] += 1
    state["pos"] = result.content
    print("正方发言  " + "*" * 80)
    print(state["pos"])
    return state


def neger(state):
    prompt = """你是一位辩论高手且是一位小红书忠实女权主义用户，现在在辩论中担任反方，根据【论题】和【正方发言】，用口语化的风格说出你的观点，驳斥【正方发言】，证明【论题】的错误。500字以内。
        【论题】
        """ + state["topic"] + """
        【正方发言】
        """ + state["pos"]
    result = llm.invoke(prompt)
    state["neg"] = result.content
    print("反方发言  " + "*" * 80)
    print(state["neg"])
    return state


def judgement(state):
    if state["count"] >= 2:
        return "finish"
    else:
        return "proceed"

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("poser", poser)
    graphBuilder.add_node("neger", neger)

    graphBuilder.add_edge(START, "poser")
    graphBuilder.add_edge("poser", "neger")
    graphBuilder.add_conditional_edges("neger", judgement, {"finish": END, "proceed": "poser"})

    graph = graphBuilder.compile()
    return graph


if __name__ == "__main__":
    graph = buildGraph()

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {"topic": "30岁的男人应该有多少钱积蓄", "count": 0}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result)