from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from utils.config import Config, ModelConfig, ModelProvider
from typing import Optional
import os

def create_llm(model_config: ModelConfig = Config.MODEL) -> BaseChatModel:

    if model_config.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            max_tokens=Config.OLLAMA_CONTEXT_WINDOW
        )
    elif model_config.provider == ModelProvider.GROQ:
        return ChatGroq(
            model=model_config.name,
            temperature=model_config.temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_config.provider}")