# RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with Clean Architecture principles, featuring both online and offline operation modes with comprehensive testing and CI/CD pipeline.

## 🚀 Features

- **Clean Architecture Implementation**: Domain-driven design with clear separation of concerns
- **Dual Operation Modes**: Online mode with LLM-based retrieval and offline mode with exact matching
- **Conversational AI**: Stateful conversations with full history management
- **Tool Integration**: Smart document retrieval using LangChain tools with factory and strategy patterns
- **Web Interface**: Interactive Gradio-based UI for seamless user experience
- **Production Ready**: 100% test coverage, comprehensive CI/CD, type checking, and linting

## 🏗️ Architecture

The system follows Clean Architecture principles with four distinct layers:

### Domain Layer (`src/rag_system/domain/`)
Core business logic and abstract interfaces:
- `BaseAgent` - Main orchestration interface
- `BaseLLMClient` - Language model abstraction
- `BaseQuestionClassifier` - Question classification interface
- `BaseAnswerGenerator` - Response generation interface
- `BaseConversationManager` - Conversation state management
- `BaseDocumentLoader` - Document loading abstraction

### Use Cases (`src/rag_system/use_cases/`)
Application-specific business rules:
- `Agent` - Main conversation orchestrator with tool integration
- `LLMClient` - Concrete LLM implementation using LangChain
- `AnswerGenerator` - Response generation with OpenAI message format
- `tools/documents_retrival/` - Document retrieval system with:
  - Factory pattern for strategy selection
  - `ExactMatchRetrievalStrategy` for offline mode
  - `LLMRetrievalStrategy` for online mode with tool calling

### Infrastructure (`src/rag_system/infrastructure/`)
External service integrations:
- `InMemoryConversationManager` - Stateful conversation storage
- `OfflineDocumentLoader` - Local document management
- External API integrations and data persistence

### Interfaces (`src/rag_system/interfaces/`)
External interfaces and adapters:
- `GradioUI` - Web-based user interface
- CLI and API endpoints (future implementations)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd rag_system

# Install dependencies
uv sync

# Run the application
uv run main.py

# Run in offline mode (no internet required)
uv run main.py --offline
```

## 💻 Usage

### Web Interface
Start the application and navigate to the provided Gradio URL to interact with the RAG system through a user-friendly web interface.

### Operation Modes

**Online Mode** (default):
- Uses LLM-based document retrieval with tool calling
- Intelligent document selection based on query analysis
- Full conversational AI capabilities

**Offline Mode**:
- Exact match retrieval strategy
- No internet connection required
- Faster response times for predefined queries

### Running Standalone Modules

```bash
# Run agent module directly
uv run -m rag_system.use_cases.agent

# Run LLM client module
uv run -m rag_system.infrastructure.llm_client
```

## 🧪 Testing

The project maintains 100% test coverage with both unit and integration tests.

### Test Structure
```
tests/
├── unit/                           # Unit tests (7 files)
│   ├── infrastructure/
│   └── use_cases/
│       └── tools/documents_retrival/strategies/
└── integration/                    # Integration tests (1 file)
    └── test_gradio_ui.py          # Full app deployment with browser automation
```

### Integration Testing
The integration test suite includes comprehensive end-to-end testing:
- **Full Application Deployment**: Deploys the complete Gradio app in test environment
- **Browser Automation**: Uses Playwright to simulate real user interactions
- **Message Exchange Testing**: Sends messages and validates responses end-to-end
- **CI Environment**: Runs headlessly in GitHub Actions for automated validation
- **Local Development**: Opens Chromium browser for visual debugging and development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest --cov

# Run only unit tests
uv run pytest tests/unit --cov=src --cov-fail-under=100

# Run only integration tests
uv run pytest tests/integration

# Generate coverage report
uv run coverage report -m
```

## 🔧 Development

### Code Quality Tools

```bash
# Type checking
uv run mypy src
uv run mypy tests

# Linting and formatting
uv run ruff check src
uv run ruff format src

# Run all quality checks (same as CI)
uv run ruff check src && uv run ruff format --check src && uv run mypy src && uv run mypy tests
```

### Configuration
- **Ruff**: Configured for Python 3.12+ with comprehensive rule set (pycodestyle, pyflakes, isort, flake8-bugbear, pydocstyle, pyupgrade)
- **MyPy**: Strict type checking for both source and test code
- **Pytest**: Configured with coverage reporting and 100% coverage requirement

## 🚦 CI/CD Pipeline

GitHub Actions workflow (`test_and_lint.yaml`) ensures code quality:

- **Multi-branch**: Runs on all pushes and pull requests
- **Python 3.12**: Latest Python version support
- **Comprehensive Checks**:
  - Ruff linting and formatting verification
  - MyPy type checking for source and tests
  - Pytest with 100% coverage requirement for both unit and integration tests
- **Fail-fast**: Stops on first failure for quick feedback

## 🔄 Integration & Models

### LLM Integration
- **Primary**: OpenAI GPT models via LangChain
- **Direct API**: OpenAI client for advanced features
- **Message Format**: OpenAI-style conversation format
- **Tool Calling**: LangChain tools for document retrieval

### Supported Models
- GPT-4 and GPT-3.5-turbo series
- Configurable model selection via environment variables
- Extensible architecture for additional model providers

## 📦 Dependencies

### Core Runtime
- **langchain/langchain-openai**: LLM integration and tool calling
- **gradio**: Modern web UI framework
- **pydantic**: Data validation and type safety
- **openai**: Direct OpenAI API access
- **dotenv**: Environment configuration

### Development & Testing
- **pytest/pytest-cov**: Testing framework with coverage
- **mypy**: Static type analysis
- **ruff**: Fast Python linter and formatter
- **pylint**: Additional code quality checks
- **playwright**: End-to-end testing capabilities

## 🤝 Contributing

1. Ensure all tests pass: `uv run pytest --cov`
2. Maintain 100% test coverage
3. Follow type annotations and pass MyPy checks
4. Use Ruff for code formatting and linting
5. Write integration tests for new features

## 📄 License

[Add your license information here]

## 🔗 Links

- [Clean Architecture Principles](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [LangChain Documentation](https://docs.langchain.com/)
- [Gradio Documentation](https://gradio.app/docs/)