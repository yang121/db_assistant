import re

text = """
To create a table and insert data into it, you can use the following SQL commands. Since the question doesn't specify the table structure or the data to be inserted, I'll create a simple example table called `products` and insert some sample data into it.

```sql
CREATE TABLE products (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL,
    "category" TEXT NOT NULL,
    "price" REAL NOT NULL
);

INSERT INTO products ("name", "category", "price") VALUES ('Smartphone', 'Electronics', 699.99);
INSERT INTO products ("name", "category", "price") VALUES ('Headphones', 'Electronics', 149.99);
INSERT INTO products ("name", "category", "price") VALUES ('Coffee Maker', 'Home Appliances', 89.99);
```
"""

sql_pattern = re.compile(
    r"CREATE.*?;|SELECT.*?;|INSERT.*?;|UPDATE.*?;|DELETE.*?;",
    re.DOTALL
)

checked_sql_command = sql_pattern.findall(text)

print(checked_sql_command)