import os
import time
import base64
from PIL import Image
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import Runnable

# Configurable variables for application settings
MODEL_VLM = ChatOllama(model="llava")
MODEL_EMBEDDINGS = embeddings.OllamaEmbeddings(model='nomic-embed-text')
PROMPT = "List 10 technologies you can observe in the image." #Describe the image in detail, including all technologies. Provide as much information as possible based on the visual data and other given information."

IMAGES_FOLDER = r'C:\workspace\exoskeleton_images_local' # images to be processd: "png" and "annotations" -subfolders will be added
THUMBNAIL_SIZE = 512
RAG_FILES = ["rag/enriched_technologies_area51.csv"]
PROMPT = "List all the technologies provided in the context information"

def encode_file_to_base64(file_path):
    """Load a file from the file path and encodes it in base64."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def vector_store_from_files(files, model_embeddings):
    """Build a vector store retriever from a list of files."""

    # Create list of URLs from a list of files
    urls = []
    for file in files:
        urls.append("http://127.0.0.1:8088/" + file) # Local web server is required: run_web_server.bat

    # Build a vector store from the URLs
    return vector_store_from_urls(urls, model_embeddings)

def vector_store_from_urls(urls, model_embeddings):
    """Build a vector store retriever from a list of URLs."""
    
    # Load documents from each URL and split them into chunks
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=10)
    doc_splits = text_splitter.split_documents(docs_list)

    # Build a vector store from the split documents
    vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=model_embeddings)
    return vectorstore.as_retriever()

def annotate_image(image_path, retriever):
    """Annotate an image using the provided langchain."""
    image_b64 = encode_file_to_base64(image_path)
    #return chain.invoke({"rag": chain.context, "text": chain.prompt, "image": image_b64})
    


    def prompt_func_rag(data):
        query_results = retriever.invoke(data["text"])
        rag_content = " ".join([doc.page_content for doc in query_results])
        text = data["text"]
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{image_b64}",
        }
        content_parts = [{"type": "text", "text": rag_content}, {"type": "text", "text": text}, image_part]
        return [HumanMessage(content=content_parts)]
    chain = prompt_func_rag | MODEL_VLM | StrOutputParser()
    print("annotating...")
    annotation = chain.invoke({"rag": retriever, "text": PROMPT, "image": image_b64})
    print("annotation: " + annotation)
    return annotation

def init():
    """Initialize the application and check requirements."""

    # Create suubfolders to store annotations and thumbnails
    os.makedirs(IMAGES_FOLDER + r"\annotations", exist_ok=True)
    os.makedirs(IMAGES_FOLDER + r"\thumbnails", exist_ok=True)

    # Check that all files in RAG_FILES exist
    for file in RAG_FILES:
        if not os.path.exists(file):
            print(f"ERR: RAG file {file} not found. Please provide a valid file path.")
            exit(1)

    # Check that web access to local files works
    ret = WebBaseLoader("http://127.0.0.1:8088/run_web_server.bat").load()
    if "Error" in ret[0].page_content:
        print("ERR: WebBaseLoader failed to load a file. Start run_web_server.bat and try again")
        exit(1)

def create_thumbnail(image_path, thumbnail_path):
    """Convert an image to PNG format and resize it."""
    with Image.open(image_path) as img:
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))  # resize to max width/height of 512
        img.save(thumbnail_path, "PNG")

def create_annotation(thumbnail_path, chain, annotation_path):
    """Create an annotation for an image using the provided langchain."""
    result = chain.invoke(thumbnail_path)
    with open(annotation_path, "w") as text_file:
        text_file.write(result)
        print(f"thumbnail: {thumbnail_path}\nresult: {result}")

def main():
    """Main function to control the workflow of the application."""

    # Initialize application and check requirements
    print("1 -------- Initializing application")
    init()

    # Create a vector database for RAG
    print("2 -------- Creating vector database for Retrieval Augmented Generation (RAG)")
    print(f"Using files: {RAG_FILES}...")
    start_time = time.time()
    retriever = vector_store_from_files(RAG_FILES, MODEL_EMBEDDINGS)
    print(f"...done! (elapsed time: {(time.time()-start_time):.2f} seconds)")  

    # Annotate image files as instructed by the prompt
    print("3 -------- Annotating image files")
    for filename in os.listdir(IMAGES_FOLDER):
        if filename.endswith("jpg") and filename.startswith("a000"):
            start_time = time.time() 
            base_filename = os.path.splitext(filename)[0]

            # Define paths for image, thumbnail and annotation
            image_path = os.path.join(IMAGES_FOLDER, filename)
            thumbnail_path = os.path.join(IMAGES_FOLDER + r"\thumbnails", base_filename + ".png")
            annotation_path = os.path.join(IMAGES_FOLDER + r"\annotations", base_filename + ".txt")

            # Create thumbnail if missing
            if not os.path.exists(thumbnail_path):
                create_thumbnail(image_path, thumbnail_path)

            # Create annotation if missing
            #if not os.path.exists(annotation_path):
            #    print(f"Creating {annotation_path}...")
            #    create_annotation(thumbnail_path, chain, annotation_path)
            #    print(f"...done! (elapsed time: {(time.time()-start_time):.2f} seconds)")

            #print(f"Annotating {thumbnail_path}...")
            #annotation = annotate_image(thumbnail_path, retriever)
            #print(annotation)

            image_b64 = encode_file_to_base64(thumbnail_path)

            def prompt_func_rag(data):
                query_results = retriever.invoke(data["text"])
                if query_results:
                    rag_content = " ".join([doc.page_content for doc in query_results])
                else:
                    rag_content = "No relevant content found."
                print("RAG:" + rag_content)

                text = data["text"]
                image_part = {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                }
                #content_parts = [{"type": "text", "text": rag_content}, {"type": "text", "text": text}, image_part]
                content_parts = [{"type": "text", "text": text}, image_part]
                return [HumanMessage(content=content_parts)]

            chain = prompt_func_rag | MODEL_VLM | StrOutputParser()
            print(chain.invoke({"rag": retriever, "text": PROMPT, "image": image_b64}))

if __name__ == "__main__":
    main()
