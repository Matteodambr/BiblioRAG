"""RAG agent wrapper for Paper-QA2 with Gemini integration."""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from bibliorag.config import Config

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a RAG query."""
    
    question: str
    answer: str
    context: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
    
    def __str__(self) -> str:
        """Return formatted answer string."""
        return f"Q: {self.question}\n\nA: {self.answer}"


class RAGAgent:
    """RAG agent wrapper for Paper-QA2 with Gemini integration.
    
    This class provides a high-level interface for performing RAG operations
    on documents from your Mendeley library using Paper-QA2 and Google Gemini.
    """
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the RAG agent.
        
        Args:
            config: BiblioRAG configuration. If None, will load from environment.
        """
        self.config = config or Config.from_env()
        self._docs: Optional[Any] = None
        self._settings: Optional[Any] = None
        
        # Set up Gemini API key for litellm
        if self.config.gemini.api_key:
            os.environ["GEMINI_API_KEY"] = self.config.gemini.api_key
    
    def _get_settings(self) -> Any:
        """Get Paper-QA2 settings configured for Gemini."""
        if self._settings is not None:
            return self._settings
        
        try:
            from paperqa import Settings
        except ImportError as e:
            raise ImportError(
                "paper-qa is not installed. Please install it with: "
                "pip install paper-qa"
            ) from e
        
        # Configure Gemini model via litellm
        model_name = f"gemini/{self.config.gemini.model_name}"
        
        self._settings = Settings(
            llm=model_name,
            summary_llm=model_name,
            embedding=self.config.gemini.embedding_model,
        )
        
        return self._settings
    
    async def _get_docs(self) -> Any:
        """Get or create Paper-QA2 Docs object."""
        if self._docs is not None:
            return self._docs
        
        try:
            from paperqa import Docs
        except ImportError as e:
            raise ImportError(
                "paper-qa is not installed. Please install it with: "
                "pip install paper-qa"
            ) from e
        
        self._docs = Docs()
        return self._docs
    
    async def add_documents(self, paths: Optional[list[Path]] = None) -> int:
        """Add documents to the RAG index.
        
        Args:
            paths: List of file paths to add. If None, adds all files from
                   the references directory.
                   
        Returns:
            Number of documents added.
        """
        if paths is None:
            # Add all PDFs from references directory
            if not self.config.references_dir.exists():
                logger.warning("References directory does not exist")
                return 0
            
            paths = list(self.config.references_dir.glob("*.pdf"))
        
        if not paths:
            logger.info("No documents to add")
            return 0
        
        docs = await self._get_docs()
        added = 0
        
        for path in paths:
            if not path.exists():
                logger.warning("File not found: %s", path)
                continue
            
            try:
                await docs.aadd(path)
                added += 1
                logger.info("Added document: %s", path.name)
            except Exception as e:
                logger.error("Failed to add document %s: %s", path.name, e)
        
        logger.info("Added %d documents to the index", added)
        return added
    
    async def query(self, question: str) -> QueryResult:
        """Query the document collection.
        
        Args:
            question: The question to ask.
            
        Returns:
            QueryResult with the answer and context.
        """
        docs = await self._get_docs()
        settings = self._get_settings()
        
        # Run the query
        answer = await docs.aquery(question, settings=settings)
        
        # Extract context from the answer
        context = []
        if hasattr(answer, "contexts"):
            for ctx in answer.contexts:
                context.append({
                    "text": ctx.context if hasattr(ctx, "context") else str(ctx),
                    "doc_name": ctx.doc.docname if hasattr(ctx, "doc") else "Unknown",
                    "score": ctx.score if hasattr(ctx, "score") else 0.0,
                })
        
        return QueryResult(
            question=question,
            answer=answer.answer if hasattr(answer, "answer") else str(answer),
            context=context,
            cost=answer.cost if hasattr(answer, "cost") else 0.0,
        )
    
    async def summarize(self, question: Optional[str] = None) -> QueryResult:
        """Summarize the documents in the collection.
        
        Args:
            question: Optional focus question for the summary.
            
        Returns:
            QueryResult with the summary.
        """
        if question is None:
            question = "Summarize the main findings and themes across all documents."
        
        return await self.query(question)
    
    async def find_contradictions(self) -> QueryResult:
        """Find contradictions across documents.
        
        Returns:
            QueryResult with any contradictions found.
        """
        question = (
            "Identify any contradictions, disagreements, or conflicting findings "
            "across the documents. List each contradiction with the relevant sources."
        )
        return await self.query(question)
    
    def clear(self) -> None:
        """Clear the document index."""
        self._docs = None
        logger.info("Document index cleared")
