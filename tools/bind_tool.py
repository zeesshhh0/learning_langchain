from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dotenv import load_dotenv


load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

@tool
def multiply(a:int , b: int)-> int:
  "this tool will multiply the two numbers"
  return a * b

llm_with_tool = llm.bind_tools([multiply])

resp = llm_with_tool.invoke('yooo whats upp')

print(resp)