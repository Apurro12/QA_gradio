import gradio as gr
from use_cases.classify_question import QuestionClassifier
from use_cases.answer_question import AnswerGenerator
from use_cases.agent import Agent
from infrastructure.llm_client import LLMClient

# Initialize dependencies
llm = LLMClient()
question_classifier = QuestionClassifier()
answer_generator = AnswerGenerator(llm)

# Create agent instance
agent = Agent(llm, question_classifier, answer_generator)

def respond_to_question(user_question: str) -> str:
    return agent.respond_user_question(user_question)

# Build Gradio Interface
iface = gr.Interface(
    fn=respond_to_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
    outputs="text",
    title="RAG Agent Q&A",
    description="Ask questions and get answers based on documents."
)

if __name__ == "__main__":
    iface.launch()
