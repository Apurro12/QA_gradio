from infrastructure.vector_store import VectorStore

class BaseDocumentSelector:
    def select_document(self, query: str) -> str:
        """
        Select the most relevant document based on the query.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class DocumentSelector(BaseDocumentSelector):
    def __init__(self):
        self.vector_store = VectorStore()

    def select_document(self, query: str) -> str:
        """
        Select the most relevant document based on the query.
        """
        return self.vector_store.query(query)
