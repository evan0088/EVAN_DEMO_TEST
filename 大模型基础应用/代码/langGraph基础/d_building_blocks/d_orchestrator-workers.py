from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
from langgraph.types import Send
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel,Field
import operator

class Section(BaseModel):
    num: int = Field(description="章节序号")
    name:str=Field(description="章节标题，采用“【第三章】 迷雾重现”的形式")
    description:str=Field(description="章节描述，最前面是章节标题，后面是章节内容。章节标题采用“【第三章】 迷雾重现”的形式，单独一行")

class Sections(BaseModel):
    sections: list[Section] = Field(description="文章的各个子章节")

class State(TypedDict):
    storyLine:str
    sections:list[Section]
    completedSections:Annotated[list,operator.add]
    novel:str

class WorkerState(TypedDict):
    section:Section
    completedSections:Annotated[list,operator.add]

llm = ChatOllama(model="qwen2.5:7b")

def getWholeStory(state):
    prompt = "你是一位著名小说家，正在创作一部精彩的侦探小说，首先给出故事梗概，1000字左右。"
    state["storyLine"] = llm.invoke(prompt).content
    print("storyLine  " + "*" * 80)
    print(state["storyLine"])
    return state

def orchestrate(state):
    planner = llm.with_structured_output(Sections)
    result = planner.invoke("""你是一位著名小说家，正在创作一部精彩的侦探小说，根据下面的【故事梗概】，将小说分成10个章节，并给出章节的剧情发展。1000字左右。
                            【故事梗概】
                            """+state["storyLine"])
    state["sections"] = result.sections
    print("sections  " + "*" * 80)
    for section in state["sections"]:
        print(str(section.name+'        '+section.description))
    return state

def work(state:WorkerState):
    prompt = """你是一位著名小说家，正在创作一部精彩的侦探小说，根据下面提供的【章节标题】和【章节概述】，完成其中的一个章节，最前面是章节标题，后面是章节内容。章节标题采用“【第三章】 迷雾重现”的形式，单独一行。章节的序号为"""+str(state["section"].num)+""",本章节长度1500字左右
    
    【章节标题】
    """+state["section"].name+"""
    
    【章节概述】
    """+state["section"].description
    result = llm.invoke(prompt)
    return {"num":state["section"].num,"name":state["section"].name,"description":state["section"].description,"completedSections":[{"num":state["section"].num,"content":result.content}]}

def synthesizer(state):
    completedSections = sorted(state["completedSections"],key=lambda completeSection:completeSection["num"])
    novel = "\n\n".join([completeSection["content"] for completeSection in completedSections])
    state["novel"] = novel
    return state

def assignWorkers(state:State):
    return [Send("work",{"section":s}) for s in state["sections"]]

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getWholeStory", getWholeStory)
    graphBuilder.add_node("orchestrate", orchestrate)
    graphBuilder.add_node("work", work)
    graphBuilder.add_node("synthesizer", synthesizer)

    graphBuilder.add_edge(START, "getWholeStory")
    graphBuilder.add_edge("getWholeStory", "orchestrate")
    graphBuilder.add_conditional_edges("orchestrate",assignWorkers,["work"])
    graphBuilder.add_edge("work", "synthesizer")

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()

    from courseCode import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result["novel"])