from pydantic import BaseModel, Field
from langchain.tools import tool

class MultiplyInput(BaseModel):
  "this tool multiplies two Numbers"
  a: int = Field(description="first number to multiply") 
  b: int = Field(description="second number to multiply") 
  
@tool("multiplication", args_schema=MultiplyInput, return_direct=True)
def multiply(a, b):
  return a * b


print(multiply.invoke({'a':10, 'b': 102}))