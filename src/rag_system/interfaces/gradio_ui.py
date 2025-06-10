import gradio as gr
from rag_system.domain.agent import BaseAgent


# This Agent class should be modified to an abstract class
def launch_gradio(agent: BaseAgent):
    def respond_to_question(user_question: str) -> str:
        return agent.respond_user_question(user_question)

    iface = gr.Interface(
        fn=respond_to_question,
        inputs=gr.Textbox(lines=2, placeholder="Ask your question here..."),
        outputs="text",
        title="RAG Agent Q&A",
        description="Ask questions and get answers based on documents."
    )

    iface.launch()

if __name__ == "__main__":
    from rag_system.interfaces.strategies_classifier.classify_question import ExactMatchClassifier
    from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.infrastructure.llm_client import OfflineLLMClient

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    from rag_system.use_cases.agent import Agent

    agent = Agent(question_classifier, answer_generator)

    # Launch the Gradio interface
    launch_gradio(agent)