from langchain_core.runnables import Runnable
from a_gather_infomation import getInfomation
from typing import Dict, Optional

class CustomModel(Runnable):
    def __init__(self, model_endpoint: str, api_key: str):
        self.endpoint = model_endpoint
        self.api_key = api_key

    def invoke(self, prompt: str, modelName:str="deepseek-ai/DeepSeek-V3",maxTokens:int=10000,tempreture=0.6,config: Optional[Dict] = None) -> Dict:
        """调用自定义模型API"""
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model":modelName,
            "messages": [{"role": "user", "content": prompt}],
            "temperature":tempreture,
            "max_tokens": maxTokens
        }

        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

llm = CustomModel(
        model_endpoint="https://api.siliconflow.cn/v1/chat/completions",
        api_key="xxxx"
    )

state = {}
state = getInfomation(state)
prompt = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【竞品分析】和【女性汽车市场发展趋势】，结合你自己的思考，完成一份新车型设计方案。该方案输出的形式要遵循【章节安排】，全文在10000字左右。

         【女性汽车市场发展趋势】
        """ + state["femaleMarketTrend"] + """

        【竞品分析】

        """ + state["competitorStatus"] + """

        【章节安排】
        第一章 新车型的设计背景和理念 #这一章节里必须给新车型起一个符合女性汽车市场的名字
        第二章 市场趋势分析
        第三章 用户人群分析与使用场景
        第四章 技术方案和细分市场定制
        第五章 竞品分析与差异化竞争优势
        第六章 产品卖点
        第七章 定价策略
        第八章 海外市场
        第九章 销量预估"""

result = llm.invoke(prompt)

print(result)


