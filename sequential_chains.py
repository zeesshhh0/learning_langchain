from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt_research = PromptTemplate(
  template='you are a senior researcher on the topic of cricket, research on {topic} of 200 words',
  input_variables= ['topic']
)

prompt_summerize = PromptTemplate(
  template='write me five short and important points from this text: {text}, points should be around 10 words each',
  input_variables= ['text']
)

parser = StrOutputParser()

chain = prompt_research | llm | parser | prompt_summerize | llm | parser

response = chain.invoke({'topic': 'the impact of MS Dhoni'})

chain.get_graph()

# print(response)

