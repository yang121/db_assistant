import dotenv
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
import requests
import os
from langchain_openai import OpenAIEmbeddings

# 设置API密钥和基础URL环境变量
# API_KEY = os.getenv("CUSTOM_API_KEY", "<Your Key>")
dotenv.load_dotenv()

class ChatSiliconFlow(LLM):
    model: str = 'deepseek-ai/DeepSeek-V3'
    use_stream: bool = False
    max_tokens: int = 1024
    temperature: int = 1.0
    top_p: float = 0.7
    top_k: int = 50
    frequency_penalty: float = 0.5
    n: int = 1
    api_key: str
    base_url: str = 'https://api.siliconflow.cn/v1/chat/completions'

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def llm_completions(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "use_stream": self.use_stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        print(payload, headers)
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()  # 检查响应状态码
        except requests.exceptions.HTTPError as e:
            print("HTTP 错误:", e)
            print("响应内容:", response.text)  # 打印 API 返回的错误信息
            raise
        return response.json()["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: list = None, model: str = 'deepseek-ai/DeepSeek-V3') -> str:
        response = self.siliconflow_completions(model=self.model, prompt=prompt)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response

class SiliconFlowEmbeddings(OpenAIEmbeddings):
    base_url: str = 'https://api.siliconflow.cn/v1/embeddings'
    api_key: str = os.environ["SILICONFLOW_API_KEY"]
    model: str = 'BAAI/bge-large-zh-v1.5'

    def embeddings(self, model: str, prompt: str) -> str:
        payload = {
            "model": "BAAI/bge-large-zh-v1.5",
            "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!",
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.base_url, json=payload, headers=headers)

        print(response.text)




if __name__ == "__main__":
    llm = ChatSiliconFlow(model="deepseek-ai/DeepSeek-V3", api_key=os.environ["SILICONFLOW_API_KEY"])
    response = llm.invoke("你好")
    print(response)

