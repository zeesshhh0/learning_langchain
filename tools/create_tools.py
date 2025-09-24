from langchain.tools import tool
from pydantic import BaseModel, Field


@tool
def multiply(a: int, b:int) -> int:
  "multiplies two numbers"
  return a * b


result = multiply.invoke({'a': 12, 'b': 10})

print(result)