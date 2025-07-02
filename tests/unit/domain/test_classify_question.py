import pytest

from rag_system.domain.classify_question import BaseQuestionClassifier
from rag_system.domain.document import Document, EmptyResponse


class TestBaseQuestionClassifier:
    def test_base_question_classifier_is_abstract(self):
        """Test that BaseQuestionClassifier cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseQuestionClassifier(None)  # type: ignore

    def test_base_question_classifier_abstract_method(self):
        """Test that classify method is abstract."""
        class IncompleteClassifier(BaseQuestionClassifier):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteClassifier(None)  # type: ignore

    def test_base_question_classifier_concrete_implementation(self):
        """Test that a concrete implementation works correctly."""
        class ConcreteClassifier(BaseQuestionClassifier):
            def __init__(self, document_loader):
                super().__init__(document_loader)
            
            def classify(self, question: str) -> Document | EmptyResponse:
                return EmptyResponse(content=None)

        classifier = ConcreteClassifier(None)
        result = classifier.classify("test question")
        assert result == EmptyResponse(content=None)