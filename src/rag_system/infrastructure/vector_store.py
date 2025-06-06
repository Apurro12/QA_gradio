from typing import List
from domain.document import Document

class VectorStore:
    def __init__(self):
        self.documents = []

    def add_document(self, document: Document):
        self.documents.append(document)

    def query(self, query: str) -> str:
        # Dummy implementation: return the first document
        return self.documents[0].content if self.documents else "No documents found."