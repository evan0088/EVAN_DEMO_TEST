import os

from openai import OpenAI

class deepseek_client:
    def __init__(self):
        self.client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com")


    def invoke(self,prompt):
        return self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}}
        ).choices[0].message