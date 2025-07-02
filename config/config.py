"""Configuration management for RAG system."""

from enum import Enum
import json
from pydantic import BaseModel, Field


class OperationMode(str, Enum):
    """Operation modes for the RAG system."""
    ONLINE = "online"
    OFFLINE = "offline"


class LLMConfig(BaseModel):
    """LLM configuration settings."""
    
    mode: str = Field(description="llm mode offline or online")


class RetrievalConfig(BaseModel):
    """Document retrieval configuration."""
    ConnectionManager: str = Field(description="Connect to load the data local storage/api/database/etc")
    QueryService: str = Field(description="Query service to run queries agains the connection")
    QueryServiceWargs: dict[str, object] = Field( # type: ignore
        description="Additional arguments for the query service",
    )
class UIConfig(BaseModel):
    """UI configuration settings."""
    
    interface: str = Field(description="UI interface type")
    host: str = Field(description="Host address")
    port: int = Field(description="Port number")
    share: bool = Field(description="Enable public sharing")


class RAGConfig(BaseModel):
    """Main RAG system configuration."""
    
    mode: OperationMode
    llm: LLMConfig
    retrieval: RetrievalConfig
    ui: UIConfig


def load_config(config_path: str) -> RAGConfig:
    """Load configuration from JSON file."""

    with open(config_path) as f:
        config_data = json.load(f)

    return RAGConfig(**config_data)

