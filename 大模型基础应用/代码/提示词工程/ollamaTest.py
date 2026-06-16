from ollama import chat

messages = [
  {
    'role': 'user',
    'content': '你在国内最喜欢是哪座山？',
  },
]

response = chat('qwen2.5:7b', messages=messages,options={"temperature":0.0})
print(response['message']['content'])