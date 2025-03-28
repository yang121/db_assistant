
# 导入dotenv库，用于从.env文件加载环境变量
import dotenv
import os

from langchain_openai import ChatOpenAI

# 导入SQLDatabase工具，用于与SQL数据库进行交互
from langchain_community.utilities.sql_database import SQLDatabase
# 导入SQLDatabaseChain，用于创建一个结合了语言模型和数据库的处理链
from SQLChainFix import SQLChain
from langchain_siliconflow import SiliconFlow
from langchain.prompts import PromptTemplate

# 加载.env文件中的环境变量
dotenv.load_dotenv()

# 从指定的数据库URI创建SQL数据库实例，此处使用的是SQLite数据库
# db = SQLDatabase.from_uri("sqlite:///chinook.db")
db = SQLDatabase.from_uri("oracle+cx_oracle://MES:mes_2024@172.17.193.237:1521/mesdev")
print(db.dialect)
print(db.get_context())

#
# 创建OpenAI模型实例，设置temperature为0（完全确定性输出），并启用详细日志记录
# llm = ChatOpenAI(model='deepseek-chat', api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")

# SCNET
# DeepSeek-R1-Distill-Qwen-7B
# DeepSeek-R1-Distill-Qwen-32B
# DeepSeek-R1-Distill-Llama-70B
# DeepSeek-R1-671B
# QwQ-32B
# llm = ChatOpenAI(
#     model='QwQ-32B',
#     api_key=os.environ['SCNET_API_KEY'],
#     base_url="https://api.scnet.cn/api/llm/v1",
#     streaming=True
# )

# # 硅基流动版
llm = SiliconFlow(model="deepseek-ai/DeepSeek-V3", api_key=os.environ["SILICONFLOW_API_KEY"])

# 创建SQL数据库链，结合了语言模型和数据库，用于处理基于数据库的查询
template = PromptTemplate(
    # input_variables=["input", "table_info", "dialect"],
    template="""
    请使用中文回答问题
    """
)
db_chain = SQLChain.from_llm(
    llm,
    db,
    verbose=True,
    # prompt=template,
)

# 使用数据库链运行查询，此处查询“有多少员工？”
# print(db_chain.invoke("帮我查一下订单表, 并告诉我大致情况"))
print(db_chain.invoke("帮我计算一下每个人的销售额"))