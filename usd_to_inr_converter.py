from langchain.chat_models import init_chat_model
from dotenv import load_dotenv, get_key
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain.tools import tool
import requests
import os

load_dotenv()

llm = init_chat_model(model='gemini-2.5-flash', model_provider='google_genai')

prompt = ChatPromptTemplate([
  ('system', "you are a currency converter bot user will ask you to convert the indian rupees into us dollers"),
  ('human', 'i have {amount} rupees, convert them')
])

@tool
def convert_rupees_to_usd(amount: int)-> float:
  "this tool can convert rupees into usd, it takes rupees amount and returns the usd amount"
  try:
    url = f"https://openexchangerates.org/api/latest.json?app_id={os.getenv('OPENEXCHANGE_API_KEY')}&base=USD&symbols=INR&prettyprint=false&show_alternative=false"
    response = requests.get(url)
    if response.status_code == 200:
      data = response.json()
      result = round(amount / data['rates']['INR'], 2)
      return result
    else:
      print(f"Error: {response.status_code} - {response.text}")

  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
  return False

llm_with_tool = llm.bind_tools([convert_rupees_to_usd])

messages = prompt.format_messages(amount=100)
tool_called = llm_with_tool.invoke(messages)

messages.append(tool_called)

tool_call = tool_called.tool_calls[0]
tool_result = convert_rupees_to_usd.invoke(tool_call['args'])
messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call['id']))

final_result= llm_with_tool.invoke(messages)
print(final_result.content)