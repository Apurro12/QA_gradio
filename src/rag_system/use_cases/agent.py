from typing import List, Optional
from rag_system.domain.agent import BaseAgent
from rag_system.domain.answer_generator import BaseAnswerGenerator
from rag_system.domain.conversation_manager import BaseConversationManager, Message
from rag_system.interfaces.strategies_classifier.exact_match_classifier import BaseQuestionClassifier

class Agent(BaseAgent):
    def __init__(
            self, 
            question_classifier: BaseQuestionClassifier,
            answer_generator: BaseAnswerGenerator,
            conversation_manager: Optional[BaseConversationManager] = None
            ):
        self.question_classifier = question_classifier
        self.answer_generator = answer_generator
        self.conversation_manager = conversation_manager

    def respond_user_question(self, question: str, history: List[Message]) -> str:
        """
        Respond to a question by classifying it, selecting a document, and generating an answer.
        History is not currently used, it is here to comply with gradio interface.
        """
        document = self.question_classifier.classify(question)
        context = document["content"]

        # This is a placeholder for more sophisticated context retrieval logic
        if not context:
            return "No relevant information found to answer your question."

        response = self.answer_generator.generate(question, context)

        return response
    
    def chat(self, message: str, history: List[Message]) -> str:
        """
        Handle a chat message with conversation context.
        If conversation manager is not provided, fallback to simple question answering.
        If there is not a valid documents/context for the question, do not add it to the prompt.
        History is not currently used, it is here to comply with gradio interface.
        Maybe it should be used to maintain conversation history in the future.
        """
        if not self.conversation_manager:
            return self.respond_user_question(message, history)
        
        # Combine conversation context with document context
        full_context = ""
        if history:
            context: List[str] = []
            for msg in history:
                context.append(f"{msg.role}: {msg.content}")
            conversation_context = "\n".join(context)
            del context # I know this is not needed, but I want to be explicit

            full_context += f"Conversation History:\n{conversation_context}\n \n"
        
        document = self.question_classifier.classify(message)
        document_context = document["content"]        
        if document_context:
            full_context += f"Relevant Information:\n{document_context}"
        
        response = self.answer_generator.generate(message, full_context)

        if self.conversation_manager:
            self.conversation_manager.update_conversation(
                history + [Message(role="user", content=message), Message(role="assistant", content=response)]
            )

        return response

if __name__ == "__main__":  # pragma: no cover, JUST DO WHEN RUNNING THIS FILE DIRECTLY
    from rag_system.interfaces.strategies_classifier.exact_match_classifier import ExactMatchClassifier
    from rag_system.use_cases.answer_generator import OfflineAnswerGenerator
    from rag_system.infrastructure.document_loader import OfflineDocumentLoader
    from rag_system.infrastructure.llm_client import OfflineLLMClient
    from rag_system.infrastructure.conversation_manager import InMemoryConversationManager

    llm = OfflineLLMClient()
    offline_document_loader = OfflineDocumentLoader()
    question_classifier = ExactMatchClassifier(document_loader=offline_document_loader)
    answer_generator = OfflineAnswerGenerator(llm=llm)
    conversation_manager = InMemoryConversationManager()
    
    agent = Agent(question_classifier, answer_generator, conversation_manager)
    
    # Interactive chat example
    print("Chat interface started. Type 'quit' to exit, 'clear' to clear conversation.")
    history = [Message(role="assistant", content='empty message')]
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = agent.chat(user_input, history)
        print(f"Assistant: {response}")
        print()
        print("----------------")
        print("----------------")
        print("----------------")
        print()