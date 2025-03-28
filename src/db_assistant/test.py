

# 导入SQLDatabase工具，用于与SQL数据库进行交互
from langchain_community.utilities.sql_database import SQLDatabase

# 从指定的数据库URI创建SQL数据库实例，此处使用的是SQLite数据库
# db = SQLDatabase.from_uri("sqlite:///chinook.db")
db = SQLDatabase.from_uri("oracle+cx_oracle://MES:mes_2024@172.17.193.237:1521/mesdev")
print(db.dialect)
print('--------------------------')

print(db.get_table_names)
print('--------------------------')

print(db.get_usable_table_names())
print('--------------------------')

