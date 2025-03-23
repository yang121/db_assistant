# 导入dotenv库，用于从.env文件加载环境变量
import dotenv
import os
import getpass
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()
llm = ChatOpenAI(model='deepseek-chat', api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

# llm = GoogleGenerativeAI(model=os.environ['MODEL'], google_api_key=os.environ['GEMINI_API_KEY'], temperature=0, verbose=True)
print(llm.invoke('今天天气怎么样'))