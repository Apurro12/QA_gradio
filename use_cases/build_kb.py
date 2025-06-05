from infrastructure.vector_store import VectorStore
from domain.document import Document

def build_kb(documents):
    """
    Build the knowledge base by storing documents in the vector store.
    """
    vector_store = VectorStore()
    for doc in documents:
        vector_store.add_document(Document(content=doc))
    print("Knowledge base built successfully.")