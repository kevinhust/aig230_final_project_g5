import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
KB_DIR = "kb"
CHROMA_PATH = "data/chroma_db"
MODEL_NAME = "BAAI/bge-m3"

def build_vector_store():
    print(f"Loading documents from {KB_DIR}...")
    loader = DirectoryLoader(KB_DIR, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Initializing embedding model: {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'mps' for Mac M1/M2 if available, but cpu is safer for first run
    )
    
    print(f"Creating ChromaDB at {CHROMA_PATH}...")
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    # Chroma in recent versions persists automatically or via client.
    print("Vector store build complete.")
    return db

if __name__ == "__main__":
    if not os.path.exists(KB_DIR):
        print(f"Error: {KB_DIR} directory not found.")
    else:
        build_vector_store()
