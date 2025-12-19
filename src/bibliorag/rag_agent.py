"""RAG agent wrapper for Paper-QA2 with Gemini integration."""
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bibliorag.config import Config

logger = logging.getLogger(__name__)


def _compute_documents_fingerprint(references_dir: Path) -> str:
    """Compute a fingerprint hash of all PDFs in the references directory.
    
    The fingerprint includes:
    - List of PDF filenames (sorted)
    - File sizes
    - Last modification times
    
    Returns:
        SHA256 hash of the fingerprint data.
    """
    if not references_dir.exists():
        return hashlib.sha256(b"").hexdigest()
    
    pdf_files = sorted(references_dir.glob("*.pdf"))
    
    # Build fingerprint data
    fingerprint_parts = []
    for pdf in pdf_files:
        stat = pdf.stat()
        fingerprint_parts.append(f"{pdf.name}|{stat.st_size}|{stat.st_mtime}")
    
    fingerprint_str = "\n".join(fingerprint_parts)
    return hashlib.sha256(fingerprint_str.encode()).hexdigest()


def _load_document_metadata(state_file: Path) -> dict[str, dict[str, Any]]:
    """Load document metadata from sync state file.
    
    Returns a mapping from filename to document metadata (title, authors, year).
    """
    if not state_file.exists():
        return {}
    
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        metadata = {}
        for doc_id, doc_data in data.get("documents", {}).items():
            for file_path in doc_data.get("files", []):
                filename = Path(file_path).name
                metadata[filename] = {
                    "title": doc_data.get("title", "Unknown"),
                    "authors": doc_data.get("authors", []),
                    "year": doc_data.get("year"),
                }
        return metadata
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load document metadata: %s", e)
        return {}


def _format_citation(filename: str, metadata: dict[str, dict[str, Any]]) -> str:
    """Format a citation using title and authors if available.
    
    Args:
        filename: The PDF filename.
        metadata: Mapping from filename to document metadata.
        
    Returns:
        Formatted citation string (e.g., "Smith et al. (2023) - Title of Paper").
    """
    if filename not in metadata:
        return filename
    
    doc_meta = metadata[filename]
    title = doc_meta.get("title", "Unknown")
    authors = doc_meta.get("authors", [])
    year = doc_meta.get("year")
    
    # Format authors - handle both 'last_name' and 'surname' field names
    # Mendeley API uses 'last_name' and 'first_name'
    def get_last_name(author: dict) -> str:
        """Extract last name from author dict, handling different field names."""
        return author.get("last_name") or author.get("surname") or author.get("name", "Unknown")
    
    if authors:
        if len(authors) == 1:
            author_str = get_last_name(authors[0])
        elif len(authors) == 2:
            author_str = f"{get_last_name(authors[0])} & {get_last_name(authors[1])}"
        else:
            author_str = f"{get_last_name(authors[0])} et al."
    else:
        author_str = "Unknown"
    
    # Build citation
    if year:
        return f"{author_str} ({year}) - {title}"
    else:
        return f"{author_str} - {title}"


@dataclass
class QueryResult:
    """Result from a RAG query."""
    
    question: str
    answer: str
    context: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    _doc_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    
    def __str__(self) -> str:
        """Return formatted answer string."""
        return f"Q: {self.question}\n\nA: {self.answer}"
    
    def get_citations(self) -> list[str]:
        """Get list of unique citations from context with titles and authors."""
        citations = []
        seen = set()
        for ctx in self.context:
            doc_name = ctx.get("doc_name", "Unknown")
            if doc_name not in seen:
                seen.add(doc_name)
                # Format citation with title and authors
                formatted = _format_citation(doc_name, self._doc_metadata)
                citations.append(formatted)
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
        self._doc_metadata: dict[str, dict[str, Any]] = {}
        
        # Cache-related attributes
        self._cache_file: Path = self.config.cache_dir / "docs_index.pkl"
        self._fingerprint_file: Path = self.config.cache_dir / "fingerprint.txt"
        
        # Set up API keys and URLs for litellm
        if self.config.llm.provider == "gemini" and self.config.llm.api_key:
            os.environ["GOOGLE_API_KEY"] = self.config.llm.api_key
        
        # Set up Ollama API base URL for litellm (for both LLM and embeddings)
        if self.config.llm.provider == "ollama" or self.config.embedding.provider == "ollama":
            os.environ["OLLAMA_API_BASE"] = self.config.embedding.ollama_url
        
        # Suppress verbose LiteLLM logging (errors, retries, etc.)
        self._configure_litellm_logging()
    
    def _configure_litellm_logging(self) -> None:
        """Configure LiteLLM to use less verbose logging."""
        import logging
        
        # Suppress LiteLLM's verbose INFO logs
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
        
        # Keep paper-qa client warnings but suppress verbose ones
        logging.getLogger("paperqa.clients").setLevel(logging.ERROR)
    
    def _load_doc_metadata(self) -> None:
        """Load document metadata from sync state for citation formatting."""
        self._doc_metadata = _load_document_metadata(self.config.state_file)
    
    def _save_index_cache(self) -> None:
        """Save the current Docs index to cache."""
        if self._docs is None:
            return
        
        try:
            self.config.ensure_directories()
            
            # Save the Docs object
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._docs, f)
            
            # Save the current fingerprint
            fingerprint = _compute_documents_fingerprint(self.config.references_dir)
            self._fingerprint_file.write_text(fingerprint)
            
            logger.info("Saved index cache to %s", self._cache_file)
        except Exception as e:
            logger.warning("Failed to save index cache: %s", e)
    
    def _load_index_cache(self) -> bool:
        """Load the Docs index from cache if valid.
        
        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        # Check if cache files exist
        if not self._cache_file.exists() or not self._fingerprint_file.exists():
            logger.info("No index cache found")
            return False
        
        try:
            # Check if fingerprint matches (documents haven't changed)
            cached_fingerprint = self._fingerprint_file.read_text().strip()
            current_fingerprint = _compute_documents_fingerprint(self.config.references_dir)
            
            if cached_fingerprint != current_fingerprint:
                logger.info("Document fingerprint changed, cache invalid")
                return False
            
            # Load the cached Docs object
            with open(self._cache_file, "rb") as f:
                self._docs = pickle.load(f)
            
            logger.info("Loaded index from cache")
            print("✓ Loaded index from cache (documents unchanged)\n")
            return True
        except Exception as e:
            logger.warning("Failed to load index cache: %s", e)
            return False
    
    def clear_cache(self) -> None:
        """Clear the index cache."""
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
            if self._fingerprint_file.exists():
                self._fingerprint_file.unlink()
            logger.info("Cleared index cache")
            print("✓ Index cache cleared")
        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)
    
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
        if not self.config.mendeley.client_id or not self.config.mendeley.client_secret:
            logger.warning("Mendeley credentials not configured, skipping sync")
            return 0, 0
        
        try:
            client = self._get_mendeley_client()
            
            # Check if authenticated and connection is valid
            try:
                client.ensure_authenticated()
            except RuntimeError as e:
                logger.error(str(e))
                print(f"\n{e}")
                return 0, 0
            
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
        """Get Paper-QA2 settings configured for Gemini and embedding model."""
        if self._settings is not None:
            return self._settings
        
        try:
            from paperqa import Settings
        except ImportError as e:
            raise ImportError(
                "paper-qa is not installed. Please install it with: "
                "pip install paper-qa"
            ) from e
        
        # Configure model based on provider
        if self.config.llm.provider == "gemini":
            model_name = f"gemini/{self.config.llm.model_name}"
        elif self.config.llm.provider == "ollama":
            model_name = f"ollama/{self.config.llm.model_name}"
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm.provider}")
        
        # Configure embedding based on provider
        embedding_config = self.config.embedding
        if embedding_config.provider == "ollama":
            # Use Ollama with nomic-embed-text or other local model
            embedding = f"ollama/{embedding_config.model_name}"
        elif embedding_config.provider == "google":
            # Use Google embedding model
            embedding = embedding_config.model_name
        else:
            # Default to Ollama
            embedding = f"ollama/{embedding_config.model_name}"
        
        logger.info(f"Creating Settings with LLM: {model_name}, embedding: {embedding}")
        
        # Configure LiteLLM Router with fallback support
        # Build model list with primary model and optional fallback
        model_list = [
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": model_name,
                    "num_retries": 2,  # Retry primary model 2 times before fallback
                    "timeout": 60,  # 60 second timeout per request
                }
            }
        ]
        
        # Add fallback model only if using Gemini and fallback is configured
        if self.config.llm.provider == "gemini" and self.config.llm.fallback_model:
            fallback_model = self.config.llm.fallback_model
            logger.info(f"Configuring fallback model: {fallback_model}")
            model_list.append({
                "model_name": model_name,  # Use same name for routing
                "litellm_params": {
                    "model": fallback_model,
                    "timeout": 120,  # Local models may need more time
                }
            })
        
        llm_config = {
            "model_list": model_list,
            "num_retries": 3,
            "retry_after": 5,  # Wait 5 seconds before retrying
            "fallbacks": [model_name] if (self.config.llm.provider == "gemini" and self.config.llm.fallback_model) else [],
            "context_window_fallbacks": [model_name] if (self.config.llm.provider == "gemini" and self.config.llm.fallback_model) else [],
        }
        
        self._settings = Settings(
            llm=model_name,
            summary_llm=model_name,
            llm_config=llm_config,
            summary_llm_config=llm_config,
            embedding=embedding,
            temperature=self.config.paperqa.temperature,
        )
        
        # Configure agent LLM to use Gemini instead of defaulting to GPT-4
        self._settings.agent.agent_llm = model_name
        self._settings.agent.agent_llm_config = llm_config
        self._settings.agent.search_count = self.config.paperqa.search_count
        
        # Configure parsing enrichment LLM - can use different model for indexing
        # This allows using fast local models (e.g., ollama/llama3.2:1b) during indexing
        # while keeping powerful models (Gemini) for query answering
        enrichment_model = self.config.llm.enrichment_model or model_name
        
        # If enrichment model is different from main model, create separate config
        if enrichment_model != model_name:
            enrichment_llm_config = {
                "model_list": [
                    {
                        "model_name": enrichment_model,
                        "litellm_params": {
                            "model": enrichment_model,
                            "num_retries": 3,
                            "timeout": 60,
                        }
                    }
                ],
                "num_retries": 3,
                "retry_after": 5,
            }
            logger.info(f"Using separate enrichment LLM for indexing: {enrichment_model}")
        else:
            enrichment_llm_config = llm_config
        
        self._settings.parsing.enrichment_llm = enrichment_model
        self._settings.parsing.enrichment_llm_config = enrichment_llm_config
        
        # Configure answer quality settings
        self._settings.answer.answer_max_sources = self.config.paperqa.answer_max_sources
        self._settings.answer.evidence_k = self.config.paperqa.evidence_k
        self._settings.answer.answer_length = self.config.paperqa.answer_length
        self._settings.answer.evidence_summary_length = self.config.paperqa.evidence_summary_length
        self._settings.answer.max_concurrent_requests = self.config.paperqa.max_concurrent_requests
        
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
        
        # Try to load from cache first
        if self._load_index_cache():
            self._documents_added = True
            return self._docs
        
        # Get settings first so Docs uses the configured LLMs
        settings = self._get_settings()
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
        
        from tqdm import tqdm
        
        docs = await self._get_docs()
        settings = self._get_settings()
        added = 0
        
        print(f"Adding {len(paths)} documents to index...")
        
        # Use tqdm with leave=True to keep the bar visible, and position=0 for proper updating
        for path in tqdm(paths, desc="Indexing documents", unit="doc", leave=True, position=0, dynamic_ncols=True):
            if not path.exists():
                logger.warning("File not found: %s", path)
                continue
            
            try:
                await docs.aadd(path, settings=settings)
                added += 1
                logger.info("Added document: %s", path.name)
            except Exception as e:
                logger.error("Failed to add document %s: %s", path.name, e)
                tqdm.write(f"⚠ Failed to add {path.name}: {e}")
        
        print(f"\n✓ Successfully indexed {added}/{len(paths)} documents\n")
        logger.info("Added %d documents to the index", added)
        
        # Save the index cache after adding documents
        if added > 0:
            print("Saving index cache...")
            self._save_index_cache()
            print("✓ Index cache saved\n")
        
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
        from tqdm import tqdm
        
        # Auto-sync from Mendeley if enabled and not already synced
        if self.auto_sync and not self._synced:
            logger.info("Auto-syncing references from Mendeley...")
            self.sync_references()
            self._synced = True
        
        # Load document metadata for citation formatting
        self._load_doc_metadata()
        
        # Add documents if not already added
        docs = await self._get_docs()
        if not self._documents_added:
            print("Indexing documents...")
            await self.add_documents()
            self._documents_added = True
        
        settings = self._get_settings()
        
        # Print query progress information
        print(f"\nProcessing query: {question}")
        print(f"Evidence chunks to retrieve: {settings.answer.evidence_k}")
        print(f"Estimated LLM calls: {settings.answer.evidence_k + 1}")
        print("\nThis may take 30-60 seconds...\n")
        
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
            _doc_metadata=self._doc_metadata,
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
