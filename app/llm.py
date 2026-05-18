from langchain_ollama import ChatOllama

from app.config import MODEL_NAME, MODEL_TEMPERATURE


def get_interpreter_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE
    )


def get_query_generator_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE
    )