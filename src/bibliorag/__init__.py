"""BiblioRAG - RAG wrapper for Mendeley library using Paper-QA2 and Gemini."""

__version__ = "0.1.0"

from bibliorag.mendeley_client import MendeleyClient
from bibliorag.rag_agent import RAGAgent
from bibliorag.config import Config

__all__ = ["MendeleyClient", "RAGAgent", "Config", "__version__"]
