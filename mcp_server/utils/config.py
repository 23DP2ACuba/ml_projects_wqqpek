from dataclasses import dataclass
from enum import Enum

class ModelProvider(str, Enum):
    OLLAMA='ollama'
    GROQ="groq"
    
@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider
    
QWEN_2_5 = ModelConfig("qwen2.5", temperature=0.0, provider=ModelProvider.OLLAMA)
LLAMA_3_3 = ModelConfig("llama-3.3-70b-versatile", temperature=0.0, provider=ModelProvider.GROQ)

class Config:
    SEED = 42
    MODEL = QWEN_2_5
    OLLAMA_CONTEXT_WINDOW = 4096
    
    class Server:
        HOST = "127.0.0.1"
        PORT = 8080
        SSE_PATH = "/sse"
        TRANSPORT = "sse"
        
    class Agent:
        MAX_ITERATIONS = 10
        
