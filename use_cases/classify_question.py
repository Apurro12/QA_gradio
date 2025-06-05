from infrastructure.vector_store import VectorStore

def classify_question(question):
    """
    Classify the question and retrieve the most relevant document.
    """
    vector_store = VectorStore()
    result = vector_store.query(question)
    return result