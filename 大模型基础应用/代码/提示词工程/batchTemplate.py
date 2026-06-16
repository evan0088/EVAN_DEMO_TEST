from ollama import chat

testSamples = """2023-02-15，寓意吉祥的节日，股票佰笃[BD]美股开盘价10美元，虽然经历了波动，但最终以13美元收盘，成交量微幅增加至460,000，投资者情绪较为平稳。
2023-04-05，市场迎来轻松氛围，股票盘古(0021)开盘价23元，尽管经历了波动，但最终以26美元收盘，成交量缩小至310,000，投资者保持观望态度。""".splitlines()

promptTemplate = """{}抽取上面这段文本中的日期、股票名称、开盘价、收盘价、成交量这5类实体，以下列形式输出：
{{'日期': ['2023-01-10'], '股票名称': ['古哥-D[EOOE]'], '开盘价': ['100美元'],  '收盘价': ['102美元'], 成交量': ['520000']}}，不允许输出其他字符。"""

def predictOneTemplate(inputText):
    messages = [
      {
        'role': 'user',
        'content': promptTemplate.format(inputText),
      },
    ]

    response = chat('qwen2.5:7b', messages=messages,options={"temperature":0.0})
    return response['message']['content']



def predictBatchTemplate(testSamples):
    for sample in testSamples:
        result = predictOneTemplate(sample)
        print("*"*40)
        print(result)

if __name__ == '__main__':
    predictBatchTemplate(testSamples)