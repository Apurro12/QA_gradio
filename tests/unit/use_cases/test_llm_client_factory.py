from rag_system.use_cases.llm_client import LLMClient, OfflineLLMClient
from rag_system.use_cases.llm_client_factory import LLM_CLIENT_CLASS_MAP


class TestLLMClientFactory:
    def test_llm_client_class_map_contains_expected_classes(self):
        """Test that the LLM client class map contains the expected classes."""
        assert "offline" in LLM_CLIENT_CLASS_MAP
        assert "llm" in LLM_CLIENT_CLASS_MAP
        assert LLM_CLIENT_CLASS_MAP["offline"] == OfflineLLMClient
        assert LLM_CLIENT_CLASS_MAP["llm"] == LLMClient

    def test_llm_client_class_map_keys(self):
        """Test that the LLM client class map has the expected keys."""
        expected_keys = {"offline", "llm"}
        assert set(LLM_CLIENT_CLASS_MAP.keys()) == expected_keys

    def test_llm_client_class_map_values_are_classes(self):
        """Test that the LLM client class map values are classes."""
        for client_class in LLM_CLIENT_CLASS_MAP.values():
            assert isinstance(client_class, type)