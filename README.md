# RAG Project (LangChain + ChromaDB)

This is a Retrieval-Augmented Generation (RAG) project built with [LangChain](https://python.langchain.com/) and [ChromaDB](https://www.trychroma.com/). It allows loading PDF documents, splitting and vectorizing them for storage, and then using an AI Agent to answer user questions based on the document content.

## ✨ Features

- **Document Loading**: Supports loading PDF documents (defaults to `src/public/python-tutorial.pdf`).
- **Document Splitting**: Uses `RecursiveCharacterTextSplitter` for intelligent splitting of long documents.
- **Vector Storage**: Uses ChromaDB for local persistent storage of document vectors.
- **Embedding Model**: Integrates OpenAI Embeddings for text vectorization.
- **Intelligent Retrieval**: Builds a RAG pipeline to retrieve relevant document fragments based on user queries.
- **AI Agent**: An intelligent dialogue assistant capable of retrieving knowledge before answering questions.
- **LangGraph Integration**: Includes LangGraph configuration to support building complex agent workflows.

## 📂 Project Structure

```plaintext
RAG/
├── langgraph.json          # LangGraph configuration file
├── pyproject.toml          # Project metadata and build configuration
├── requirements.txt        # Python dependency list
├── src/                    # Source code directory
│   ├── agent.py            # Main entry point for the AI Agent logic
│   ├── constants.py        # Global constants and environment variable configuration
│   ├── model_factory.py    # Factory for LLM and Embedding models
│   ├── rag_workfolw.py     # Core RAG pipeline (build & retrieval test)
│   ├── db/                 # Vector database operations
│   │   └── chroma.py       # ChromaDB initialization and configuration
│   ├── loader/             # Document loaders
│   │   └── pdf_loader.py   # PDF file loading logic
│   ├── public/             # Public resources (PDF files and vector store data)
│   ├── retriever/          # Retriever construction logic
│   ├── splitter/           # Document splitting logic
│   └── tools/              # LangChain Tools definitions
│       └── retriever_tool.py # Retrieval tool used by the Agent
└── .env                    # Environment variable configuration file (create this yourself)
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure your system has Python 3.12+ installed.

```bash
# Clone the repository
git clone <repository_url>
cd RAG
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv
# Activate virtual environment (Windows)
.\venv\Scripts\activate
# Activate virtual environment (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root and add your API Key:

```ini
# .env file
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# If using Zhipu AI
ZHIPUAI_API_KEY=your_zhipuai_key
# Enable LangSmith tracing (Optional)
LANGSMITH_TRACING=true
```

### 4. Test RAG Pipeline

You can run `src/rag_workfolw.py` to test the full process of document loading, splitting, vectorization, and retrieval.

```bash
python src/rag_workfolw.py
```

_This script will load the default PDF, build the vector index, and demonstrate a simple query retrieval._

### 5. Run AI Agent

Run `src/agent.py` to start the intelligent Q&A assistant.

```bash
python src/agent.py
```

_This script initializes the Agent and demonstrates an automated response to a preset question ("What are the application areas of Python?")._

## 🛠️ Core Modules

- **rag_workfolw.py**: Defines the `build_pipeline` function, orchestrating the complete process from PDF loading to generating the Retriever.
- **model_factory.py**: Encapsulates methods `get_chat_model` (for LLM) and `get_embedding_model` (for Embeddings) to unify model configuration.
- **tools/retriever_tool.py**: Defines `@tool` `retrieve_context`, a LangChain tool allowing the Agent to call the RAG pipeline for knowledge when needed.
- **agent.py**: Uses `create_agent` to create an Agent with tool-calling capabilities. The system prompt restricts it to answering questions based only on retrieved content.

## 📝 Notes

- **PDF Path**: The default PDF path is configured in `DEFAULT_PDF_PATH` within `src/constants.py`. Ensure a valid PDF file exists at that path or modify the code to point to your file.
- **LangGraph Config**: The `langgraph.json` file currently points to `src/agents/agent_4.py:agent`. If you plan to use LangGraph CLI, ensure the path matches your actual code file (e.g., `src/agent.py`).

## 🤝 Contribution

Issues and Pull Requests are welcome to improve this project!
