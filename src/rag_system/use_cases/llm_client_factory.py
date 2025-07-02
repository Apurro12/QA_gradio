from rag_system.domain.llm_client import BaseLLMClient
from rag_system.use_cases.llm_client import LLMClient, OfflineLLMClient


LLM_CLIENT_CLASS_MAP: dict[str, type[BaseLLMClient]] = {
    "offline": OfflineLLMClient,
    "llm": LLMClient
}