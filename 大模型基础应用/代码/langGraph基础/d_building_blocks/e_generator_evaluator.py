from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict,Literal
from pydantic import BaseModel,Field

class State(TypedDict):
    topic:str
    article:str
    feedback:str
    qualified:str
    count:int

class Feedback(BaseModel):
    grade:Literal['合格','不合格'] = Field(description="判断文章的逻辑性是否合格")
    feedback:str = Field(description="修改意见的具体内容")

llm = ChatOllama(model="qwen2.5:7b")

def generate(state):
    if state.get("feedback"):
        prompt = """根据提供的主题写一篇论证文章。确保文章逻辑严密、有说服力。
        主题为"""+state["topic"]+"""
        同时你需要考虑如下的修改建议："""+state["feedback"]
    else:
        prompt = """根据提供的主题写一篇论证文章。确保文章逻辑严密、有说服力。
                主题为""" + state["topic"]
        prompt = """你是一位小学生，完全不会写作文，现在需要你写一篇论证文章。
                 主题为""" + state["topic"]
    result = llm.invoke(prompt)
    state["count"] += 1
    state["article"] = result.content
    print("generate  " + "*" * 80)
    print(state["article"])
    return state

def evaluate(state):
    prompt = """判断【论证文章】是否很好地论证了【主题】，是否逻辑严密，有说服力。如果合格，指出文章的优点；如果不合格，给出具体的修改意见。
    
    【主题】
    """+state["topic"]+"""
    
    【论证文章】
    """+state["article"]

    evaluator = llm.with_structured_output(Feedback)
    result = evaluator.invoke(prompt)
    state["qualified"] = result.grade
    state["feedback"] = result.feedback
    print("evaluate  " + "*" * 80)
    print(state["qualified"])
    print(state["feedback"])
    return state

def judgement(state):
    if state["count"] >= 2:
        return "accept"
    else:
        if state["qualified"] == "合格":
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