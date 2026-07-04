from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict

import showGraph
from deepseek_connection import deepseek_client


class State(TypedDict):
    topic: str
    aspect: str
    faction: str
    debate: str


# llm = ChatOllama(model="qwen2.5:7b")
llm = deepseek_client()


def getFraction(state):
    prompt = """你是一位国学大师，你认为“""" + state["topic"] + """”这个观点是 """ + state[
        "aspect"] + """ 的，那么在["儒家","法家","道家"]这三个学派中，你最可能持哪一派的观点？只允许使出学派的名称，不允许输出其他字符。"""
    state["faction"] = llm.invoke(prompt).content
    print("faction  " + "*" * 80)
    print(state["faction"])
    return state


def selectFraction(state):
    if state["faction"] == "儒家":
        return "Confucian"
    elif state["faction"] == "法家":
        return "Legalists"
    elif state["faction"] == "道家":
        return "Taoism"


def getDebateFromConfucian(state):
    prompt = """你是一位国学大师，你认为“""" + state["topic"] + """”这个观点是 """ + state[
        "aspect"] + """ 的，这是儒家的观点，请使用儒家思想对这个问题详细展开论证。"""
    state["debate"] = llm.invoke(prompt).content
    print(state["debate"])
    return state


def getDebateFromLegalists(state):
    prompt = """你是一位国学大师，你认为“""" + state["topic"] + """”这个观点是 """ + state[
        "aspect"] + """ 的，这是法家的观点，请使用法家思想对这个问题详细展开论证。"""
    state["debate"] = llm.invoke(prompt).content
    print(state["debate"])
    return state


def getDebateFromTaoism(state):
    prompt = """你是一位国学大师，你认为“""" + state["topic"] + """”这个观点是 """ + state[
        "aspect"] + """ 的，这是道家的观点，请使用道家思想对这个问题详细展开论证。"""
    state["debate"] = llm.invoke(prompt).content
    print(state["debate"])
    return state


def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getFraction", getFraction)
    graphBuilder.add_node("getDebateFromConfucian", getDebateFromConfucian)
    graphBuilder.add_node("getDebateFromLegalists", getDebateFromLegalists)
    graphBuilder.add_node("getDebateFromTaoism", getDebateFromTaoism)

    graphBuilder.add_edge(START, "getFraction")
    graphBuilder.add_conditional_edges("getFraction", selectFraction,
                                       {"Confucian": "getDebateFromConfucian", "Legalists": "getDebateFromLegalists",
                                        "Taoism": "getDebateFromTaoism"})

    graph = graphBuilder.compile()
    return graph


if __name__ == "__main__":
    graph = buildGraph()

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {"topic": "地法天，天法道，道法自然", "aspect": "正确"}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result)
