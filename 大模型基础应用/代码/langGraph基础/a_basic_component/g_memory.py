from langgraph.graph import StateGraph, START,END
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOllama(model="qwen2.5:7b")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("chatbot", chatbot)

    graphBuilder.add_edge(START, "chatbot")
    graphBuilder.add_edge("chatbot",END)

    graph = graphBuilder.compile()
    return graph

def buildGraphWithMemory():
    graphBuilder = StateGraph(State)
    #实例化memory
    memory = MemorySaver()

    graphBuilder.add_node("chatbot", chatbot)

    graphBuilder.add_edge(START, "chatbot")
    graphBuilder.add_edge("chatbot",END)

    #编译图的时候指定memory
    graph = graphBuilder.compile(checkpointer=memory)
    return graph

def singleRound(graph):
    while True:
        try:
            userInput = input("User: ")
            if userInput.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            state = {"messages": [{"role": "user", "content": userInput}]}
            result = graph.invoke(state)
            # print(result)
            print("Assitant:", result["messages"][-1].content)
        except Exception as e:
            print("发生错误："+str(e))

def multiRound(graph,config):
    while True:
        try:
            userInput = input("User: ")
            if userInput.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            state = {"messages": [{"role": "user", "content": userInput}]}
            result = graph.invoke(state, config=config)
            print(result)
            print("Assitant:", result["messages"][-1].content)
        except Exception as e:
            print("发生错误："+str(e))

if __name__ == "__main__":
    graph = buildGraph()
    # singleRound(graph)

    graphWithMemory = buildGraphWithMemory()
    config = {"configurable": {"thread_id": "1"}}
    multiRound(graphWithMemory,config)