from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
import json

class State(TypedDict):
    topic:str
    article:str
    feedback:str
    qualified:str
    count:int


llm = ChatOllama(model="qwen2.5:7b")

def generate(state):
    if state.get("feedback"):
        prompt = """根据提供的主题写一篇论证文章。确保文章逻辑严密、有说服力。
        主题为"""+state["topic"]+"""
        同时你需要考虑如下的修改建议："""+state["feedback"]
    else:
        #prompt = """根据提供的主题写一篇论证文章。确保文章逻辑严密、有说服力。
        #        主题为""" + state["topic"]
        prompt = """你是一位小学生，完全不会写作文，现在需要你写一篇论证文章。
                 主题为""" + state["topic"]
    result = llm.invoke(prompt)
    state["count"] += 1
    state["article"] = result.content
    print("generate  " + "*" * 80)
    print(state["article"])
    return state

def evaluate(state):
    prompt = """判断【论证文章】是否很好地论证了【主题】，是否逻辑严密，有说服力。如果不合格，给出具体的修改意见，并按照【指定格式】输出，【指定格式】中的“是否合格”只允许输出“是”或者“否”。只允许输出【指定格式】规定的字符，不允许输出任何其他字符。
    
    【指定格式】
    {"是否合格":"", "修改意见":""}
    
    【主题】
    """+state["topic"]+"""
    
    【论证文章】
    """+state["article"]

    result = llm.invoke(prompt)
    print("evaluate  " + "*" * 80)
    print(result.content)
    resultJson = json.loads(result.content)

    state["qualified"] = resultJson["是否合格"]
    state["feedback"] = resultJson["修改意见"]
    return state

def judgement(state):
    if state["count"] >= 5:
        return "accept"
    else:
        if state["qualified"] == "是":
            return "accept"
        else:
            return "reject"

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("generate", generate)
    graphBuilder.add_node("evaluate", evaluate)

    graphBuilder.add_edge(START, "generate")
    graphBuilder.add_edge("generate", "evaluate")
    graphBuilder.add_conditional_edges("evaluate",judgement, {"accept":END,"reject":"generate"})

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    from courseCode import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {"topic":"日本经济会在未来再次崛起","count":0}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result)