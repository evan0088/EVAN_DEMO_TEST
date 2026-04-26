from hanlp_restful import HanLPClient
# 需要安装 pip install hanlp_restful

# 实例化HanLPClient对象
HanLP = HanLPClient(url='https://www.hanlp.com/api',auth=None,language='zh')

# 初始化文本
content = "鲁迅, 浙江绍兴人, 五四新文化运动的重要参与者, 代表作朝花夕拾。"

# msra: 一种命名实体识别的规范
result = HanLP.parse(text=content, tasks=['ner/msra'])
print('result--->', result)