from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader(file_path='docs/Zishan_resume.pdf')

docs = pdf.load()

print(docs)