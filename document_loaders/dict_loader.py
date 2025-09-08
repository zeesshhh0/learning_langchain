from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


dict_loader = DirectoryLoader(
  path='docs',
  glob='*.pdf',
  loader_cls=PyPDFLoader
)

docs = dict_loader.load()

print(docs[0].page_content)