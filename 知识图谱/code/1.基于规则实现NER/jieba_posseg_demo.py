import jieba.posseg as pseg

text = "可在接到本决定书之日起六十日内向中国国家市场监督管理总局申请行政复议,杭州海康威视数字技术股份有限公司."

lcut = pseg.lcut(text)

for word, flag in lcut:
    print(word, flag)
