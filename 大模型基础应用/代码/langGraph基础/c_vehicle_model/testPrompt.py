from a_gather_infomation import getInfomation
from typing_extensions import TypedDict



class State(TypedDict):
    femaleMarketTrend:str
    victoriaStatus:str
    competitorStatus:str
    femaleMarketTrend:str

    basicSettings:str
    #背景，目标，理念
    chapter1:str
    #市场趋势分析
    chapter2:str
    # 用户人群分析与使用场景
    chapter3: str
    # 技术方案+细分市场定制
    chapter4: str
    # 竞品分析与差异化竞争
    chapter5: str
    #产品卖点
    chapter6: str
    #定价策略
    chapter7: str
    #海外市场
    chapter8: str
    #销量预估
    chapter9: str

    design:str

state = getInfomation({})
total = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【竞品资料】和【女性汽车市场发展趋势】，结合你自己的思考，为维多利亚品牌设计一款新车型，规划这个新车型、技术方案和定价策略。300字以内。

    【竞品资料】
    """ + state["competitorStatus"] + """

    【女性汽车市场发展趋势】
    """ + state["femaleMarketTrend"]


state["basicSettings"] = """维多利亚“精灵猫”车型规划‌
‌产品定位‌：A0级纯电都市座驾，聚焦25-35岁职场女性，主打“智能陪伴+场景化设计”。

‌核心亮点‌：

    ‌技术方案‌：
        续航升级：采用新一代磷酸铁锂电池，续航达450km（CLTC），支持15分钟快充至80%；
        安全强化：标配8气囊+女性假人优化碰撞系统，新增“夜归模式”（自动触发车内照明+远程监控）；
        智能交互：情感化语音助手支持美妆建议、生理期管理，内饰模块化设计（可选装母婴储物格/宠物箱）。

    ‌差异化设计‌：
        外饰：提供6种低饱和莫兰迪色系，车顶支持DIY投影（如节日主题动画）；
        空间：副驾配备“女神包”磁吸挂架+可冷藏口红格，后排座椅一键放平拓展购物载物空间。

‌定价策略‌：

    基础版12.8万元（对标比亚迪海鸥中配），高配版14.6万元（含全部智能配置），通过“改色胶囊”服务（3000元/次）和联名周边（如LOEWE车香）提升溢价。

‌预期竞争力‌：融合技术平权与情感化设计，填补时尚代步与家庭场景间的需求空白。"""

chapter1 = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【女性汽车市场发展趋势】和【新车型基本设定】，结合你自己的思考，完成新车型设计方案的第一章“一、设计背景和理念”，说明新车型的设计背景和理念。800字以内。

        【女性汽车市场发展趋势】
        """ + state["femaleMarketTrend"]+"""
        
        【新车型基本设定】
        
        """+state["basicSettings"]

#print(chapter1)

chapter2 = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【女性汽车市场发展趋势】、【竞品资料】和【新车型基本设定】，结合你自己的思考，完成新车型设计方案的第二章“二、市场趋势分析”，说明当前女性汽车市场的趋势。800字以内。

        【女性汽车市场发展趋势】
        """ + state["femaleMarketTrend"] + """
        
        【竞品资料】
        """ + state["competitorStatus"] + """

        【新车型基本设定】

        """ + state["basicSettings"]
#print(chapter2)

chapter3 = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【女性汽车市场发展趋势】和【新车型基本设定】，结合你自己的思考，完成新车型设计方案的第三章“三、目标客户与场景”，分析用户人群特点和产品的适用场景。800字以内。

        【女性汽车市场发展趋势】
        """ + state["femaleMarketTrend"] + """

        【新车型基本设定】

        """ + state["basicSettings"]
#print(chapter3)

chapter4 = """你是凯瑞汽车旗下维多利亚品牌的车型设计师，维多利亚品牌主要面向年轻的女性客户，根据【女性汽车市场发展趋势】和【新车型基本设定】，结合你自己的思考，完成新车型设计方案的第四章“四、技术方案”，具体说明新车型的技术方案，尤其是针对细分市场的定制特性。1500字以内。

        【女性汽车市场发展趋势】
        """ + state["femaleMarketTrend"] + """

        【新车型基本设定】

        """ + state["basicSettings"]
print(chapter4)