from dotenv import load_dotenv
from langchain_cohere import ChatCohere, create_sql_agent
from langchain_community.utilities import SQLDatabase
import os
import getpass

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatCohere(model="command-r-plus", temperature=0)
agent_executor = create_sql_agent(llm, db=db, verbose=True)
resp = agent_executor.invoke("告诉我你这个数据库存在哪里")
print('resp:', resp)
print(resp.get("output"))

