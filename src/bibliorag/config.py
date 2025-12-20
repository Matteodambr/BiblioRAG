"""Configuration management for BiblioRAG."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class MendeleyConfig:
    """Configuration for Mendeley API."""
    
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = "http://localhost:8080/callback"
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Load from environment if not set."""
        if not self.client_id:
            self.client_id = os.getenv("MENDELEY_CLIENT_ID", "")
        if not self.client_secret:
            self.client_secret = os.getenv("MENDELEY_CLIENT_SECRET", "")
        if not self.access_token:
            self.access_token = os.getenv("MENDELEY_ACCESS_TOKEN")
        if not self.refresh_token:
            self.refresh_token = os.getenv("MENDELEY_REFRESH_TOKEN")


@dataclass
class LLMConfig:
    """Configuration for LLM (Language Model)."""
    
    # Provider: "gemini" for Google Gemini API, "ollama" for local Ollama
    provider: str = "gemini"
    # Model name (e.g., "gemini-2.5-flash" for Gemini, "deepseek-r1:8b" for Ollama)
    model_name: str = "gemini-2.5-flash"
    # API key (only needed for Gemini)
    api_key: str = ""
    # Separate model for enrichment during indexing (can be local like ollama/llama3.2:1b)
    # If empty, uses main model
    enrichment_model: str = ""
    # Fallback model when Gemini hits rate limits (e.g., ollama/deepseek-r1:8b)
    # Only used when provider is "gemini". If empty, no fallback is used
    fallback_model: str = ""
    
    def __post_init__(self) -> None:
        """Load from environment if not set."""
        env_provider = os.getenv("LLM_PROVIDER", "").lower()
        if env_provider:
            self.provider = env_provider
        
        # Load model name based on provider
        if self.provider == "gemini":
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY", "")
            env_model = os.getenv("GEMINI_MODEL", "")
            if env_model:
                self.model_name = env_model
            # Only use fallback for Gemini
            env_fallback = os.getenv("FALLBACK_MODEL", "")
            if env_fallback:
                self.fallback_model = env_fallback
        elif self.provider == "ollama":
            env_model = os.getenv("OLLAMA_LLM_MODEL", "")
            if env_model:
                self.model_name = env_model
            # No fallback for Ollama - just error if it fails
            self.fallback_model = ""
        
        env_enrichment = os.getenv("ENRICHMENT_MODEL", "")
        if env_enrichment:
            self.enrichment_model = env_enrichment


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    
    # Provider: "ollama" for local, "google" for Google embeddings
    provider: str = "ollama"
    # Model name: "nomic-embed-text" for Ollama, "models/embedding-001" for Google
    model_name: str = "nomic-embed-text"
    # Ollama server URL (only used when provider is "ollama")
    ollama_url: str = "http://localhost:11434"
    
    def __post_init__(self) -> None:
        """Load from environment if not set."""
        env_provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
        if env_provider:
            self.provider = env_provider
        
        env_model = os.getenv("EMBEDDING_MODEL", "")
        if env_model:
            self.model_name = env_model
        
        env_ollama_url = os.getenv("OLLAMA_URL", "")
        if env_ollama_url:
            self.ollama_url = env_ollama_url


@dataclass
class PaperQAConfig:
    """Configuration for Paper-QA settings."""
    
    # Answer quality settings
    answer_max_sources: int = 5  # Maximum number of citations in answer
    evidence_k: int = 10  # Number of evidence chunks to retrieve
    answer_length: str = "about 200 words, but can be longer"
    evidence_summary_length: str = "about 100 words"
    
    # Performance settings
    temperature: float = 0.0  # LLM temperature (0.0 = deterministic)
    max_concurrent_requests: int = 4  # Parallel requests to LLM
    search_count: int = 8  # Number of papers to search
    
    # Document chunking settings
    split_large_pdfs: bool = True  # Split large PDFs into chapters
    large_pdf_pages: int = 100  # PDFs with more pages are considered "large"
    chunk_size_pages: int = 50  # Fallback chunk size if no outline found
    
    # Indexing settings
    use_enrichment: bool = False  # Use LLM to enrich images/tables during indexing (slower but higher quality)
    
    def __post_init__(self) -> None:
        """Load from environment if not set."""
        # Answer quality settings
        if env_val := os.getenv("PAPERQA_ANSWER_MAX_SOURCES"):
            self.answer_max_sources = int(env_val)
        if env_val := os.getenv("PAPERQA_EVIDENCE_K"):
            self.evidence_k = int(env_val)
        if env_val := os.getenv("PAPERQA_ANSWER_LENGTH"):
            self.answer_length = env_val
        if env_val := os.getenv("PAPERQA_EVIDENCE_SUMMARY_LENGTH"):
            self.evidence_summary_length = env_val
        
        # Performance settings
        if env_val := os.getenv("PAPERQA_TEMPERATURE"):
            self.temperature = float(env_val)
        if env_val := os.getenv("PAPERQA_MAX_CONCURRENT_REQUESTS"):
            self.max_concurrent_requests = int(env_val)
        if env_val := os.getenv("PAPERQA_SEARCH_COUNT"):
            self.search_count = int(env_val)
        
        # Document chunking settings
        if env_val := os.getenv("PAPERQA_SPLIT_LARGE_PDFS"):
            self.split_large_pdfs = env_val.lower() in ("true", "1", "yes")
        if env_val := os.getenv("PAPERQA_LARGE_PDF_PAGES"):
            self.large_pdf_pages = int(env_val)
        if env_val := os.getenv("PAPERQA_CHUNK_SIZE_PAGES"):
            self.chunk_size_pages = int(env_val)
        
        # Indexing settings
        if env_val := os.getenv("PAPERQA_USE_ENRICHMENT"):
            self.use_enrichment = env_val.lower() in ("true", "1", "yes")

@dataclass
class Config:
    """Main configuration for BiblioRAG."""
    
    # Sub-configurations
    mendeley: MendeleyConfig = field(default_factory=MendeleyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    paperqa: PaperQAConfig = field(default_factory=PaperQAConfig)
    
    # Paths
    references_dir: Path = field(default_factory=lambda: Path("references"))
    responses_dir: Path = field(default_factory=lambda: Path("responses"))
    cache_dir: Path = field(default_factory=lambda: Path(".bibliorag_cache"))
    state_file: Path = field(default_factory=lambda: Path(".bibliorag_state.json"))
    
    def __post_init__(self) -> None:
        """Load environment variables and ensure paths are Path objects."""
        load_dotenv()
        
        # Re-initialize sub-configs to pick up env vars
        if not self.mendeley.client_id:
            self.mendeley = MendeleyConfig()
        if not self.llm.api_key and self.llm.provider == "gemini":
            self.llm = LLMConfig()
        # Always re-initialize embedding config to pick up env vars
        self.embedding = EmbeddingConfig()
        # Always re-initialize paperqa config to pick up env vars
        self.paperqa = PaperQAConfig()
        
        # Ensure paths are Path objects
        if isinstance(self.references_dir, str):
            self.references_dir = Path(self.references_dir)
        if isinstance(self.responses_dir, str):
            self.responses_dir = Path(self.responses_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)
    
    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()
        return cls()
