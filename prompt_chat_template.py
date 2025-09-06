from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_prompt = ChatPromptTemplate(
  [('system', 'you are an customer query chatbot, answer helpfully'),
  MessagesPlaceholder('chat_history'),
  ('user', '{query}')]
)

chat_history = [HumanMessage("when i will get my refund?"), AIMessage("you will get refund by tomorrow")]

fromatted_prompt = chat_prompt.invoke({'chat_history': chat_history, 'query': "what is the status of my refund?"})

response = llm.invoke(fromatted_prompt)

print(response)