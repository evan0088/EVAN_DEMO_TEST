from langgraph.graph import StateGraph, START
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Annotated
import random

from deepseek_connection import deepseek_client
import showGraph


def updateReceiveDate(left,right):
    return max(left,right)

class State(TypedDict):
    sendDate: int
    transTime: int
    receiveDate:Annotated[int,updateReceiveDate]
    returnMessage:str

llm = deepseek_client()

def getSendData(state):
    sendData = random.randint(1,25)
    state["sendDate"] = sendData
    return state

def getTransTime(state):
    transTime = random.randint(3,5)
    state["transTime"] = transTime
    return state

def sendMessage(state):
    prompt = "你是江南机械厂的销售员王钟期，顾客张董事长在我厂预定了一批机器设备，希望能9月"+str(state["receiveDate"])+"日收到货物，厂里最早的发货日期是9月"+str(state["sendDate"])+"日，运输时间是"+str(state["transTime"])+"天，给客户发出适当的回复。"
    print(prompt)
    state["returnMessage"] = llm.invoke(prompt).content
    print("returnMessage  " + "*" * 80)
    print(state["returnMessage"])
    return state

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getSendData", getSendData)
    graphBuilder.add_node("getTransTime", getTransTime)
    graphBuilder.add_node("sendMessage", sendMessage)

    graphBuilder.add_edge(START, "getSendData")
    graphBuilder.add_edge(START, "getTransTime")
    graphBuilder.add_edge(["getSendData","getTransTime"], "sendMessage")

    graph = graphBuilder.compile()
    return graph

if __name__ == "__main__":
    graph = buildGraph()


    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {"receiveDate": 23}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result)