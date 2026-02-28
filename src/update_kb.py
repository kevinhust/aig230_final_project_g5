import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Configuration
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"

def add_to_kb(content, source="New-Entry.md", category="Misc"):
    print(f"Adding new content from {source}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    new_doc = Document(
        page_content=content,
        metadata={"source": source, "category": category}
    )
    
    db.add_documents([new_doc])
    print("Successfully added document to vector store.")

if __name__ == "__main__":
    new_content = """
    Q: Do you offer discounts for bulk orders?
    A: Yes, we offer a 10% discount for orders over 100 units. Please contact our wholesale department.
    """
    add_to_kb(new_content, source="Wholesale-Policy.md", category="Wholesale")
