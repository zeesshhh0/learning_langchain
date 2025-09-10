from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = GoogleGenerativeAI(model='gemini-2.5-flash')

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001") 

docs = [
  Document('Zeeshan is on a focused journey to become an AI Engineer.'),
  Document('Zeeshan is motivated to land a remote AI internship within the next few months.'),
  Document('Zeeshan is dedicated to building strong foundations in AI tools and technologies.'),
  Document('Zeeshan enjoys exploring practical projects that connect theory with real-world impact.'),
  Document('Zeeshan refines his skills step by step through consistent practice and learning.'),
]

vector_store = FAISS.from_documents(docs, embedding=embeddings_model)

similarity_retriever = vector_store.as_retriever(search_type='similarity', search_kwars={'k': 2})

retriever = MultiQueryRetriever.from_llm(
  retriever=similarity_retriever,
  llm=llm
)

query = "what is zeeshan doing?"

response = retriever.invoke(query)

print(response)