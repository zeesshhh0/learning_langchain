from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

joke_prompt = PromptTemplate(
  template='Write me a joke on {topic}',
  input_variables=['topic']
)

explain_prompt = PromptTemplate(
  template='explain this joke to me \n joke -> {text}',
  input_variables=['text']
)

joke_chain = joke_prompt | llm | parser

explain_chain = RunnableParallel({
  'joke': RunnablePassthrough(),
  'explain': explain_prompt | llm | parser
})

final_chain = joke_chain | explain_chain

response = final_chain.invoke({'topic': 'La la land'})

print(response)
