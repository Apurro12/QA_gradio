from use_cases.build_kb import build_kb
from use_cases.classify_question import classify_question
from use_cases.answer_question import answer_question
from use_cases.agent import LLM

def main():
    llm = LLM()

    # Step 1: Build the knowledge base
    documents = ["This is a sample document.", "Another example response."]
    build_kb(documents)

    # Step 2: Classify a question
    question = "What is an example response?"
    context = classify_question(question)

    # Step 3: Generate an answer
    answer = answer_question(question, context)
    print(f"Q: {question}\nA: {answer}")

    # Example question
    question = "What is an example response?"
    answer = llm.respond(question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()