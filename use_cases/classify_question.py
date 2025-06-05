from docs.example_docs import Document
from docs.example_docs import documents

def classify_question(question: str) -> Document:
    """
    Classify the question and retrieve the most relevant document.
    """
    for doc in documents:
        #This must be a more sophisticated search in the future
        if question in doc["questions"]:
            return doc
    return {"content": "No relevant document found.", "questions": []}