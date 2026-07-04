from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

def getTrainSchedule(queryDate,start,end):
    """根据指定的日期、起点城市名称和终点城市名称，查询列车班次"""
    print("获取列车时刻表")
    result = [["D81","12:24","14:30","北京西站","540"],["K4427","15:38","21:30","北京站","220"]]
    resultStr = "您查询的在 "+queryDate+" 这一天从 "+start+" 到 "+end+" 的列车共有 "+str(len(result))+"班：\n"
    for res in result:
        resultStr+=res[0]+" 次列车：发车时间为: "+res[1]+" 到站时间为： "+res[2]+" 发车站为： "+res[3]+" 票价为： "+res[4]+"\n"
    return resultStr

def getAvailableHotel(queryDate,location):
    """根据指定的日期和城市名称，查询可用的旅店"""
    print("获取目的地旅馆的情况")
    result = [["丽晶酒店", "五星级", "大床房", "1200"], ["丽晶大宾馆", "二星级", "标准间", "300"]]
    resultStr = "您查询的 " + queryDate + " 这一天可以预定的酒店有" + str(len(result)) + "家：\n"
    for res in result:
        resultStr += res[0] + " 等级为 " + res[1] + " 房间是 " + res[2] + " 价格是： " + res[3] + "\n"
    return resultStr

toolList = [getTrainSchedule,getAvailableHotel]

agent = create_react_agent(
            model=ChatOllama(model="qwen3.5:9b"),
            tools=toolList,
            prompt=""
        )

if __name__ == '__main__':
    prompt = "你是一位可靠的个人AI助理，我要在6月15日从北京到青岛旅游，请帮我安排一下车次和旅店。这次旅行要求尽可能舒适。"
    result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
    print(result["messages"][-1].content)