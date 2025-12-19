"""Command-line interface for BiblioRAG."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from bibliorag.config import Config
from bibliorag.mendeley_client import MendeleyClient
from bibliorag.rag_agent import RAGAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_auth_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the auth subcommand."""
    auth_parser = subparsers.add_parser(
        "auth",
        help="Authenticate with Mendeley",
        description="Initiate OAuth2 authentication with Mendeley API.",
    )
    auth_parser.set_defaults(func=cmd_auth)


def setup_sync_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the sync subcommand."""
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync references from Mendeley",
        description="Download new or updated references from Mendeley library.",
    )
    sync_parser.set_defaults(func=cmd_sync)


def setup_query_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the query subcommand."""
    query_parser = subparsers.add_parser(
        "query",
        help="Query your references",
        description="Ask questions about your references using RAG.",
    )
    query_parser.add_argument(
        "question",
        type=str,
        help="The question to ask about your references",
    )
    query_parser.set_defaults(func=cmd_query)


def setup_summarize_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the summarize subcommand."""
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize your references",
        description="Generate a summary of your references.",
    )
    summarize_parser.add_argument(
        "--focus",
        type=str,
        default=None,
        help="Optional focus question for the summary",
    )
    summarize_parser.set_defaults(func=cmd_summarize)


def setup_contradictions_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the contradictions subcommand."""
    contradictions_parser = subparsers.add_parser(
        "contradictions",
        help="Find contradictions in your references",
        description="Identify contradictions or disagreements across documents.",
    )
    contradictions_parser.set_defaults(func=cmd_contradictions)


def cmd_auth(args: argparse.Namespace) -> int:
    """Handle the auth command."""
    config = Config.from_env()
    
    if not config.mendeley.client_id or not config.mendeley.client_secret:
        print("Error: Mendeley credentials not configured.")
        print("Please set MENDELEY_CLIENT_ID and MENDELEY_CLIENT_SECRET environment variables.")
        print("\nTo get credentials:")
        print("1. Go to https://dev.mendeley.com/")
        print("2. Create a new application")
        print("3. Copy the Client ID and Client Secret")
        return 1
    
    client = MendeleyClient(config)
    auth_url = client.get_authorization_url()
    
    print("To authenticate with Mendeley:")
    print(f"\n1. Open this URL in your browser:\n   {auth_url}")
    print("\n2. Log in and authorize the application")
    print("\n3. You will be redirected to a URL. Copy the entire URL and paste it below.")
    
    callback_url = input("\nPaste the callback URL here: ").strip()
    
    if not callback_url:
        print("Error: No URL provided")
        return 1
    
    try:
        token = client.authenticate(callback_url)
        print("\nAuthentication successful!")
        print(f"\nAccess Token: {token.get('access_token', 'N/A')[:20]}...")
        print(f"Refresh Token: {token.get('refresh_token', 'N/A')[:20]}...")
        print("\nSave these tokens in your .env file:")
        print(f"MENDELEY_ACCESS_TOKEN={token.get('access_token', '')}")
        print(f"MENDELEY_REFRESH_TOKEN={token.get('refresh_token', '')}")
        return 0
    except Exception as e:
        print(f"Error during authentication: {e}")
        return 1


def cmd_sync(args: argparse.Namespace) -> int:
    """Handle the sync command."""
    config = Config.from_env()
    
    if not config.mendeley.access_token:
        print("Error: Not authenticated with Mendeley.")
        print("Run 'bibliorag auth' first to authenticate.")
        return 1
    
    client = MendeleyClient(config)
    
    try:
        updated_docs, downloaded_files = client.sync_references()
        
        if not updated_docs:
            print("No new or updated documents found.")
        else:
            print(f"\nUpdated {len(updated_docs)} documents:")
            for doc in updated_docs:
                print(f"  - {doc.title}")
            
            if downloaded_files:
                print(f"\nDownloaded {len(downloaded_files)} files to '{config.references_dir}':")
                for path in downloaded_files:
                    print(f"  - {path.name}")
        
        return 0
    except Exception as e:
        print(f"Error during sync: {e}")
        logger.exception("Sync failed")
        return 1


def cmd_query(args: argparse.Namespace) -> int:
    """Handle the query command."""
    config = Config.from_env()
    
    if not config.gemini.api_key:
        print("Error: Gemini API key not configured.")
        print("Please set GEMINI_API_KEY environment variable.")
        return 1
    
    return asyncio.run(_async_query(config, args.question))


async def _async_query(config: Config, question: str) -> int:
    """Run the query asynchronously."""
    agent = RAGAgent(config)
    
    try:
        # Add documents from references directory
        count = await agent.add_documents()
        if count == 0:
            print("Warning: No documents in the index.")
            print("Run 'bibliorag sync' first to download references.")
        
        # Run the query
        result = await agent.query(question)
        
        print(f"\nQuestion: {result.question}")
        print(f"\nAnswer:\n{result.answer}")
        
        if result.context:
            print("\nSources:")
            for ctx in result.context:
                print(f"  - {ctx.get('doc_name', 'Unknown')}")
        
        return 0
    except Exception as e:
        print(f"Error during query: {e}")
        logger.exception("Query failed")
        return 1


def cmd_summarize(args: argparse.Namespace) -> int:
    """Handle the summarize command."""
    config = Config.from_env()
    
    if not config.gemini.api_key:
        print("Error: Gemini API key not configured.")
        print("Please set GEMINI_API_KEY environment variable.")
        return 1
    
    return asyncio.run(_async_summarize(config, args.focus))


async def _async_summarize(config: Config, focus: str | None) -> int:
    """Run the summarization asynchronously."""
    agent = RAGAgent(config)
    
    try:
        # Add documents from references directory
        count = await agent.add_documents()
        if count == 0:
            print("Warning: No documents in the index.")
            print("Run 'bibliorag sync' first to download references.")
        
        # Run the summary
        result = await agent.summarize(focus)
        
        print("\nSummary:")
        print(result.answer)
        
        return 0
    except Exception as e:
        print(f"Error during summarization: {e}")
        logger.exception("Summarization failed")
        return 1


def cmd_contradictions(args: argparse.Namespace) -> int:
    """Handle the contradictions command."""
    config = Config.from_env()
    
    if not config.gemini.api_key:
        print("Error: Gemini API key not configured.")
        print("Please set GEMINI_API_KEY environment variable.")
        return 1
    
    return asyncio.run(_async_contradictions(config))


async def _async_contradictions(config: Config) -> int:
    """Find contradictions asynchronously."""
    agent = RAGAgent(config)
    
    try:
        # Add documents from references directory
        count = await agent.add_documents()
        if count == 0:
            print("Warning: No documents in the index.")
            print("Run 'bibliorag sync' first to download references.")
        
        # Find contradictions
        result = await agent.find_contradictions()
        
        print("\nContradictions Analysis:")
        print(result.answer)
        
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.exception("Contradiction analysis failed")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bibliorag",
        description="RAG wrapper for Mendeley library using Paper-QA2 and Gemini",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
    )
    
    setup_auth_parser(subparsers)
    setup_sync_parser(subparsers)
    setup_query_parser(subparsers)
    setup_summarize_parser(subparsers)
    setup_contradictions_parser(subparsers)
    
    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
