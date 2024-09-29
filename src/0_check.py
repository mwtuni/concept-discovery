from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

# Configurable variables
model_vlm = ChatOllama(model="llava")
model_embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')
image_filename = r'C:\workspace\ht\images\concept_image.png'
rag_csv_file = "rag/enriched_technologies_area51.csv"

# Check that image file exists and inform user if not
try:
    with open(image_filename, "r") as file:
        pass
except FileNotFoundError:
    print(f"ERR: Image file {image_filename} not found. Please provide a valid file path.")
print(f"Image file: {image_filename}")

# Check that the CSV file exists and inform user if not
try:
    with open(rag_csv_file, "r") as file:
        pass
except FileNotFoundError:
    print(f"ERR: CSV file {rag_csv_file} not found. Please provide a valid file path.")
print(f"CSV file: {rag_csv_file}")

# 1 Create vector database for RAG
print("Creating vector database for RAG")
urls = [
    "http://127.0.0.1:8088/" + rag_csv_file,
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=model_embedding,
)
retriever = vectorstore.as_retriever()

prompt_template = """Info about technologies and their appearance in image:
{context}
As an underground mining exoskeleton technology research expert visual language model, list all the technologies and their locations that you recognize in image: {image}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 2 Before RAG
print("Query before RAG")
chain = ( {"context": RunnablePassthrough(), "image": RunnablePassthrough()} | prompt | model_vlm | StrOutputParser() )
print(chain.invoke({"context": "Guess technologies based on generic mining exoskeleton information", "image": image_filename}))

# 3 After RAG
print("Query after RAG")
chain = ( {"context": retriever, "image": RunnablePassthrough()} | prompt | model_vlm | StrOutputParser() )
print(chain.invoke(image_filename))

# loader = PyPDFLoader("Ollama.pdf")
# doc_splits = loader.load_and_split()
