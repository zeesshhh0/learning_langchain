from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model='gemini-2.5-flash', model_provider='google_genai')

@tool
def multiply(a: int, b: int) -> int:
  "this tool will multiply two numbers"
  return a * b

llm_with_tools = llm.bind_tools([multiply])

messages = [HumanMessage("what is 20 into 13")]

mes = llm_with_tools.invoke(messages)

messages.append(mes)

tool_result = multiply.invoke(mes.tool_calls[0])

messages.append(tool_result)

final_result = llm_with_tools.invoke(messages)

print(final_result.content)