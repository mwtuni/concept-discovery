import base64
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter

"""
This example leverages the RAG (Retrieval-Augmented Generation) model to enhance the description 
of characters depicted in image by llava VLM. Specifically, it aims to provide a detailed portrayal 
of a character appearing in an image, allowing for a richer and more nuanced description compared 
to what could be achieved solely through visual observation by the llava model by default.
"""

# Configurable variables
MODEL_VLM = ChatOllama(model="llava")
MODEL_EMBEDDING = embeddings.OllamaEmbeddings(model='nomic-embed-text')
PROMPT = """
Provide a detailed description of the character in the image, paying particular attention to 
hair products and the origin of glossy sheen.
"""

def load_image(file_path):
    """Loads an image from the file path and encodes it in base64."""
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def load_and_split_documents(urls):
    """Loads documents from URLs and splits them into chunks."""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    return text_splitter.split_documents(docs_list)

def main():
    urls = ["http://127.0.0.1:8088/data.txt"]
    doc_splits = load_and_split_documents(urls)
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=MODEL_EMBEDDING)
    retriever = vectorstore.as_retriever()

    image_b64 = load_image(r'image.png')

    def prompt_func_rag(data):

        # Query the retriever for relevant rag content
        query_results = retriever.invoke(data["text"])
        if query_results:
            rag_content = " ".join([doc.page_content for doc in query_results])
        else:
            rag_content = "No relevant content found."

        # Prepare content parts including retrieved content, input text, and image
        text = data["text"]
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_b64}",
        }
        content_parts = [{"type": "text", "text": rag_content}, {"type": "text", "text": text}, image_part]

        # Return content as HumanMessage
        return [HumanMessage(content=content_parts)]

    chain = prompt_func_rag | MODEL_VLM | StrOutputParser()
    print(chain.invoke({"rag": retriever, "text": PROMPT, "image": image_b64}))

if __name__ == "__main__":
    main()
