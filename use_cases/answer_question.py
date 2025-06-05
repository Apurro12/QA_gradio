from infrastructure.openai_client import OpenAIClient

def answer_question(question: str, context: str):
    """
    Generate an answer to the question using the provided context.
    """
    client = OpenAIClient()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = client.invoke(prompt)
    return response