from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

vector_store = Chroma(
  'docs',
  GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
  'chroma_docs'
)

docs = PyPDFLoader(
  'docs/guide.pdf',
)

loaded_docs = docs.lazy_load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size=250,
  chunk_overlap=20
)

chunks = splitter.split_documents(loaded_docs)

vector_store.add_documents(chunks)

resp = vector_store.similarity_search(
  query='Database Structure Overview',
  k=2
)

print(resp)
