from rag_system.domain.agent import BaseAgent
from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.interfaces.strategies_classifier.classify_question import BaseQuestionClassifier

class Agent(BaseAgent):
    def __init__(
            self, 
            question_classifier: BaseQuestionClassifier,
            answer_generator: BaseAnswerGenerator
            ):
        self.question_classifier = question_classifier
        self.answer_generator = answer_generator

    def respond_user_question(self, question: str) -> str:
        """
        Respond to a question by classifying it, selecting a document, and generating an answer.
        """
        document = self.question_classifier.classify(question)
        context = document["content"]

        # This is a placeholder for more sophisticated context retrieval logic
        if not context:
            return "No relevant information found to answer your question."

        response = self.answer_generator.generate(question, context)

        return response

if __name__ == "__main__":
    from rag_system.interfaces.strategies_classifier.classify_question import ExactMatchClassifier
    from rag_system.use_cases.answer_question import OfflineAnswerGenerator
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.infrastructure.llm_client import OfflineLLMClient

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    
    agent = Agent(question_classifier, answer_generator)
    print(agent.respond_user_question("How can I get a refund?"))