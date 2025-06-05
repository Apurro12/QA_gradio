from infrastructure.openai_client import OpenAIClient

def answer_question(question, context):
    """
    Generate an answer to the question using the provided context.
    """
    client = OpenAIClient()
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = client.generate(prompt)
    return response