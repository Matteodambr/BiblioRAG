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
class GeminiConfig:
    """Configuration for Google Gemini API."""
    
    api_key: str = ""
    model_name: str = "gemini-1.5-pro"
    embedding_model: str = "models/embedding-001"
    
    def __post_init__(self) -> None:
        """Load from environment if not set."""
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.embedding_model:
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/embedding-001")


@dataclass
class Config:
    """Main configuration for BiblioRAG."""
    
    # Sub-configurations
    mendeley: MendeleyConfig = field(default_factory=MendeleyConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    
    # Paths
    references_dir: Path = field(default_factory=lambda: Path("references"))
    responses_dir: Path = field(default_factory=lambda: Path("responses"))
    state_file: Path = field(default_factory=lambda: Path(".bibliorag_state.json"))
    
    def __post_init__(self) -> None:
        """Load environment variables and ensure paths are Path objects."""
        load_dotenv()
        
        # Re-initialize sub-configs to pick up env vars
        if not self.mendeley.client_id:
            self.mendeley = MendeleyConfig()
        if not self.gemini.api_key:
            self.gemini = GeminiConfig()
        
        # Ensure paths are Path objects
        if isinstance(self.references_dir, str):
            self.references_dir = Path(self.references_dir)
        if isinstance(self.responses_dir, str):
            self.responses_dir = Path(self.responses_dir)
        if isinstance(self.state_file, str):
            self.state_file = Path(self.state_file)
    
    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        load_dotenv()
        return cls()
