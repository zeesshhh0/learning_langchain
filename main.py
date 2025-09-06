from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

chat_history = [
    SystemMessage(content="You are a funny helpful assistant that only reply with rhyming words."),
]

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    response = llm.invoke(chat_history)

    print(response.content)







