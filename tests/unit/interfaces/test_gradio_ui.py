from unittest.mock import MagicMock, patch
import gradio as gr
from rag_system.interfaces.gradio_ui import make_interface, make_respond_to_question, generate_session_id


class TestGradioUI:
    def test_make_respond_to_question_creates_function(self):
        """Test that make_respond_to_question creates a function."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Test response"
        
        respond_function = make_respond_to_question(mock_agent)
        
        assert callable(respond_function)

    def test_respond_to_question_calls_agent_with_message_and_history(self):
        """Test that the respond function calls agent with message and converted history."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Test response"
        
        respond_function = make_respond_to_question(mock_agent)
        
        # Mock Gradio history format
        history = [
            {"role": "user", "content": "Previous user message"},
            {"role": "assistant", "content": "Previous assistant response"}
        ]
        
        result = respond_function("Current message", history, "test-session")
        
        # Verify agent.chat was called with the message and converted history
        mock_agent.chat.assert_called_once()
        call_args = mock_agent.chat.call_args
        assert call_args[0][0] == "Current message"  # First positional arg is the message
        assert len(call_args[0][1]) == 2  # Second positional arg is the history list
        assert call_args[0][1][0].role == "user"
        assert call_args[0][1][0].content == "Previous user message"
        assert call_args[0][1][1].role == "assistant"
        assert call_args[0][1][1].content == "Previous assistant response"
        assert call_args[1]["session_id"] == "test-session"  # Check session_id keyword arg
        
        assert result == "Test response"

    def test_respond_to_question_with_empty_history(self):
        """Test that the respond function works with empty history."""
        mock_agent = MagicMock()
        mock_agent.chat.return_value = "Response to first message"
        
        respond_function = make_respond_to_question(mock_agent)
        
        result = respond_function("First message", [], "test-session")
        
        mock_agent.chat.assert_called_once_with("First message", [], session_id="test-session")
        assert result == "Response to first message"

    @patch('rag_system.interfaces.gradio_ui.gr.ChatInterface')
    def test_make_interface_creates_chat_interface(self, mock_chat_interface):
        """Test that make_interface creates a ChatInterface."""
        mock_agent = MagicMock()
        mock_interface = MagicMock()
        mock_chat_interface.return_value = mock_interface
        
        demo = make_interface(mock_agent)
        
        mock_chat_interface.assert_called_once()
        # The function returns a gr.Blocks object, not the ChatInterface directly
        assert demo is not None
        assert isinstance(demo, gr.Blocks)
        assert hasattr(demo, 'launch')
        assert callable(demo.launch)

    @patch('rag_system.interfaces.gradio_ui.gr.ChatInterface')
    def test_make_interface_configuration(self, mock_chat_interface):
        """Test that make_interface creates ChatInterface with correct configuration."""
        mock_agent = MagicMock()
        mock_interface = MagicMock()
        mock_chat_interface.return_value = mock_interface
        
        make_interface(mock_agent)
        
        # Verify ChatInterface was called with correct parameters
        call_kwargs = mock_chat_interface.call_args[1]
        assert call_kwargs["title"] == "RAG Agent Chat"
        assert call_kwargs["description"] == "Have a conversation and ask questions based on documents."
        assert call_kwargs["type"] == "messages"

class TestGenerateSessionId:
    def test_generate_session_id_returns_string(self):
        """Test that generate_session_id returns a string."""
        from gradio import Request
        
        mock_request = Request()
        session_id = generate_session_id(mock_request)
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0  # Ensure it's not an empty string