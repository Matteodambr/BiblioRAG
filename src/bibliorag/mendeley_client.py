"""Mendeley API client for fetching and syncing references."""
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from requests_oauthlib import OAuth2Session

from bibliorag.config import Config, MendeleyConfig

logger = logging.getLogger(__name__)

# Mendeley API endpoints
MENDELEY_AUTH_URL = "https://api.mendeley.com/oauth/authorize"
MENDELEY_TOKEN_URL = "https://api.mendeley.com/oauth/token"
MENDELEY_API_BASE = "https://api.mendeley.com"


@dataclass
class Document:
    """Represents a Mendeley document/reference."""
    
    id: str
    title: str
    authors: list[dict[str, str]] = field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    source: Optional[str] = None
    identifiers: dict[str, str] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    created: Optional[str] = None
    last_modified: Optional[str] = None
    
    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "Document":
        """Create a Document from Mendeley API response."""
        return cls(
            id=data.get("id", ""),
            title=data.get("title", "Untitled"),
            authors=data.get("authors", []),
            year=data.get("year"),
            abstract=data.get("abstract"),
            source=data.get("source"),
            identifiers=data.get("identifiers", {}),
            files=[],  # Files fetched separately
            created=data.get("created"),
            last_modified=data.get("last_modified"),
        )
    
    def get_content_hash(self) -> str:
        """Generate a hash of document content for change detection."""
        content = f"{self.title}|{self.year}|{self.abstract}|{self.last_modified}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SyncState:
    """Tracks synchronization state for documents."""
    
    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_sync: Optional[str] = None
    
    def save(self, path: Path) -> None:
        """Save state to file."""
        data = {
            "documents": self.documents,
            "last_sync": self.last_sync,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SyncState":
        """Load state from file."""
        if not path.exists():
            return cls()
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                documents=data.get("documents", {}),
                last_sync=data.get("last_sync"),
            )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load sync state: %s", e)
            return cls()
    
    def is_new_or_updated(self, doc: Document) -> bool:
        """Check if a document is new or has been updated."""
        if doc.id not in self.documents:
            return True
        
        stored = self.documents[doc.id]
        current_hash = doc.get_content_hash()
        return stored.get("content_hash") != current_hash
    
    def update_document(self, doc: Document, files: list[str]) -> None:
        """Update the state for a document."""
        self.documents[doc.id] = {
            "content_hash": doc.get_content_hash(),
            "title": doc.title,
            "files": files,
            "last_synced": datetime.utcnow().isoformat(),
        }


class MendeleyClient:
    """Client for interacting with the Mendeley API."""
    
    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the Mendeley client.
        
        Args:
            config: BiblioRAG configuration. If None, will load from environment.
        """
        self.config = config or Config.from_env()
        self.mendeley_config = self.config.mendeley
        self._session: Optional[OAuth2Session] = None
        self._state = SyncState.load(self.config.state_file)
    
    @property
    def session(self) -> OAuth2Session:
        """Get or create OAuth2 session."""
        if self._session is None:
            if self.mendeley_config.access_token:
                # Use existing token
                token = {
                    "access_token": self.mendeley_config.access_token,
                    "refresh_token": self.mendeley_config.refresh_token,
                    "token_type": "Bearer",
                }
                self._session = OAuth2Session(
                    client_id=self.mendeley_config.client_id,
                    token=token,
                )
            else:
                self._session = OAuth2Session(
                    client_id=self.mendeley_config.client_id,
                    redirect_uri=self.mendeley_config.redirect_uri,
                    scope=["all"],
                )
        return self._session
    
    def get_authorization_url(self) -> str:
        """Get the URL for OAuth2 authorization.
        
        Returns:
            Authorization URL to redirect user to.
        """
        authorization_url, _ = self.session.authorization_url(MENDELEY_AUTH_URL)
        return authorization_url
    
    def authenticate(self, authorization_response: str) -> dict[str, str]:
        """Complete OAuth2 authentication flow.
        
        Args:
            authorization_response: The full callback URL with auth code.
            
        Returns:
            Token dictionary with access_token and refresh_token.
        """
        # Allow insecure transport for localhost during development
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        
        token = self.session.fetch_token(
            MENDELEY_TOKEN_URL,
            authorization_response=authorization_response,
            client_secret=self.mendeley_config.client_secret,
        )
        
        self.mendeley_config.access_token = token.get("access_token")
        self.mendeley_config.refresh_token = token.get("refresh_token")
        
        return token
    
    def get_documents(self, limit: int = 500) -> list[Document]:
        """Fetch all documents from the user's library.
        
        Args:
            limit: Maximum number of documents to fetch.
            
        Returns:
            List of Document objects.
        """
        documents = []
        url = f"{MENDELEY_API_BASE}/documents"
        params = {"limit": min(limit, 500), "view": "all"}
        
        while url and len(documents) < limit:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            for item in data:
                documents.append(Document.from_api_response(item))
            
            # Handle pagination
            if "next" in response.links:
                url = response.links["next"]["url"]
                params = {}  # Params are in the URL for subsequent requests
            else:
                break
        
        logger.info("Fetched %d documents from Mendeley", len(documents))
        return documents
    
    def get_document_files(self, document_id: str) -> list[dict[str, Any]]:
        """Get files attached to a document.
        
        Args:
            document_id: Mendeley document ID.
            
        Returns:
            List of file metadata dictionaries.
        """
        url = f"{MENDELEY_API_BASE}/files"
        params = {"document_id": document_id}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def download_file(self, file_id: str, output_path: Path) -> Path:
        """Download a file from Mendeley.
        
        Args:
            file_id: Mendeley file ID.
            output_path: Path where to save the file.
            
        Returns:
            Path to the downloaded file.
        """
        url = f"{MENDELEY_API_BASE}/files/{file_id}"
        
        # Request the actual file content
        headers = {"Accept": "application/pdf"}
        response = self.session.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Downloaded file to %s", output_path)
        return output_path
    
    def sync_references(self) -> tuple[list[Document], list[Path]]:
        """Synchronize references from Mendeley.
        
        Downloads new or updated documents and their files.
        
        Returns:
            Tuple of (updated documents, downloaded file paths).
        """
        self.config.ensure_directories()
        
        documents = self.get_documents()
        updated_docs = []
        downloaded_files = []
        
        for doc in documents:
            if not self._state.is_new_or_updated(doc):
                logger.debug("Skipping unchanged document: %s", doc.title)
                continue
            
            logger.info("Processing document: %s", doc.title)
            updated_docs.append(doc)
            
            # Get and download files for this document
            files = self.get_document_files(doc.id)
            doc_files = []
            
            for file_info in files:
                file_id = file_info.get("id")
                filename = file_info.get("file_name", f"{file_id}.pdf")
                
                # Sanitize filename
                safe_filename = "".join(
                    c for c in filename if c.isalnum() or c in ".-_ "
                ).strip()
                if not safe_filename:
                    safe_filename = f"{file_id}.pdf"
                
                output_path = self.config.references_dir / safe_filename
                
                try:
                    self.download_file(file_id, output_path)
                    downloaded_files.append(output_path)
                    doc_files.append(str(output_path))
                except requests.RequestException as e:
                    logger.error("Failed to download file %s: %s", filename, e)
            
            # Update sync state
            self._state.update_document(doc, doc_files)
        
        # Save state
        self._state.last_sync = datetime.utcnow().isoformat()
        self._state.save(self.config.state_file)
        
        logger.info(
            "Sync complete: %d documents updated, %d files downloaded",
            len(updated_docs),
            len(downloaded_files),
        )
        
        return updated_docs, downloaded_files
    
    def get_sync_state(self) -> SyncState:
        """Get the current synchronization state."""
        return self._state
