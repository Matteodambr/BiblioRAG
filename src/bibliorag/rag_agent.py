"""RAG agent wrapper for Paper-QA2 with Gemini integration."""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
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
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __str__(self) -> str:
        """Return formatted answer string."""
        return f"Q: {self.question}\n\nA: {self.answer}"
    
    def get_citations(self) -> list[str]:
        """Get list of unique citations from context."""
        citations = []
        seen = set()
        for ctx in self.context:
            doc_name = ctx.get("doc_name", "Unknown")
            if doc_name not in seen:
                seen.add(doc_name)
                citations.append(doc_name)
        return citations
    
    def format_for_save(self) -> str:
        """Format the result for saving to file."""
        lines = []
        lines.append("=" * 80)
        lines.append("BIBLIORAG QUERY RESPONSE")
        lines.append("=" * 80)
        lines.append("")
        
        # Prompt
        lines.append("PROMPT:")
        lines.append("-" * 40)
        lines.append(self.question)
        lines.append("")
        
        # Model
        lines.append("MODEL:")
        lines.append("-" * 40)
        lines.append(self.model if self.model else "Unknown")
        lines.append("")
        
        # Citations
        lines.append("CITATIONS:")
        lines.append("-" * 40)
        citations = self.get_citations()
        if citations:
            for i, citation in enumerate(citations, 1):
                lines.append(f"  [{i}] {citation}")
        else:
            lines.append("  No citations available")
        lines.append("")
        
        # Response
        lines.append("RESPONSE:")
        lines.append("-" * 40)
        lines.append(self.answer)
        lines.append("")
        
        # Metadata
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.timestamp}")
        if self.cost > 0:
            lines.append(f"Cost: ${self.cost:.6f}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


class RAGAgent:
    """RAG agent wrapper for Paper-QA2 with Gemini integration.
    
    This class provides a high-level interface for performing RAG operations
    on documents from your Mendeley library using Paper-QA2 and Google Gemini.
    
    By default, automatically syncs references from Mendeley before queries
    and saves all query responses to the responses directory.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        auto_sync: bool = True,
        save_responses: bool = True,
    ) -> None:
        """Initialize the RAG agent.
        
        Args:
            config: BiblioRAG configuration. If None, will load from environment.
            auto_sync: If True, automatically sync from Mendeley before queries.
            save_responses: If True, save all query responses to files.
        """
        self.config = config or Config.from_env()
        self.auto_sync = auto_sync
        self.save_responses = save_responses
        self._docs: Optional[Any] = None
        self._settings: Optional[Any] = None
        self._mendeley_client: Optional[Any] = None
        self._synced: bool = False
        self._documents_added: bool = False
        
        # Set up Gemini API key for litellm
        if self.config.gemini.api_key:
            os.environ["GEMINI_API_KEY"] = self.config.gemini.api_key
    
    def _get_mendeley_client(self) -> Any:
        """Get or create Mendeley client for auto-sync."""
        if self._mendeley_client is None:
            from bibliorag.mendeley_client import MendeleyClient
            self._mendeley_client = MendeleyClient(self.config)
        return self._mendeley_client
    
    def sync_references(self) -> tuple[int, int]:
        """Sync references from Mendeley.
        
        Returns:
            Tuple of (updated_docs_count, downloaded_files_count).
        """
        if not self.config.mendeley.access_token:
            logger.warning("Mendeley not authenticated, skipping sync")
            return 0, 0
        
        try:
            client = self._get_mendeley_client()
            updated_docs, downloaded_files = client.sync_references()
            logger.info(
                "Synced %d documents, downloaded %d files",
                len(updated_docs),
                len(downloaded_files),
            )
            return len(updated_docs), len(downloaded_files)
        except Exception as e:
            logger.error("Failed to sync from Mendeley: %s", e)
            return 0, 0
    
    def _save_response(self, result: QueryResult) -> Path:
        """Save a query result to a file.
        
        Args:
            result: The query result to save.
            
        Returns:
            Path to the saved file.
        """
        self.config.ensure_directories()
        
        # Create filename from the result's timestamp (reuse for consistency)
        # Parse the ISO format and convert to filename-friendly format
        try:
            ts = datetime.fromisoformat(result.timestamp)
            timestamp = ts.strftime("%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a short slug from the question
        slug = "".join(c for c in result.question[:30] if c.isalnum() or c == " ")
        slug = slug.replace(" ", "_").lower()
        filename = f"{timestamp}_{slug}.txt"
        
        filepath = self.config.responses_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result.format_for_save())
        
        logger.info("Saved response to %s", filepath)
        return filepath
    
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
        
        Automatically syncs from Mendeley if auto_sync is enabled (first time only),
        and saves the response if save_responses is enabled.
        
        Args:
            question: The question to ask.
            
        Returns:
            QueryResult with the answer and context.
        """
        # Auto-sync from Mendeley if enabled and not already synced
        if self.auto_sync and not self._synced:
            logger.info("Auto-syncing references from Mendeley...")
            self.sync_references()
            self._synced = True
        
        # Add documents if not already added
        docs = await self._get_docs()
        if not self._documents_added:
            await self.add_documents()
            self._documents_added = True
        
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
        
        result = QueryResult(
            question=question,
            answer=answer.answer if hasattr(answer, "answer") else str(answer),
            context=context,
            cost=answer.cost if hasattr(answer, "cost") else 0.0,
            model=self.config.gemini.model_name,
        )
        
        # Save response if enabled
        if self.save_responses:
            self._save_response(result)
        
        return result
    
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
