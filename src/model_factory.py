from langchain import rate_limiters
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from constants import OPENAI_API_KEY


def get_llm(model: str = "gpt-4o", temperature: float = 0.7) -> ChatOpenAI:
    """Return a ChatOpenAI instance with sensible defaults."""

    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        timeout=30,
        max_tokens=1000,
    )


def get_chat_model(
    model: str = "gpt-4o", temperature: float = 0.5
):  # v1.0 新写法
    """Return the v1.0 style chat model initializer."""

    return init_chat_model(
        model=model,
        model_provider="openai",
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        timeout=10,
        max_tokens=1000,
        # rate_limiters=...
    )


def get_embedding_model(model: str = "text-embedding-3-large") -> OpenAIEmbeddings:
    """Return the embedding model used by vector store and retriever."""

    return OpenAIEmbeddings(model=model)
