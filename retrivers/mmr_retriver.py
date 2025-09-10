from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
  Document('Zeeshan is on a focused journey to become an AI Engineer.'),
  Document('Zeeshan is motivated to land a remote AI internship within the next few months.'),
  Document('Zeeshan is dedicated to building strong foundations in AI tools and technologies.'),
  Document('Zeeshan enjoys exploring practical projects that connect theory with real-world impact.'),
  Document('Zeeshan refines his skills step by step through consistent practice and learning.'),
]

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = FAISS.from_documents(docs, embeddings)

retriver = vector_store.as_retriever(
  search_type="mmr",
  search_kwargs={'k': 2, 'lamda_mult': 0.7}
)

response = retriver.invoke('what will zeeshan do next?')

print(response)