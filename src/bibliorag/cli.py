"""Command-line interface for BiblioRAG."""
import argparse
import asyncio
import logging
import sys
from typing import Any

from bibliorag.config import Config
from bibliorag.mendeley_client import MendeleyClient
from bibliorag.rag_agent import RAGAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_auth_parser(subparsers: Any) -> None:
    """Set up the auth subcommand."""
    auth_parser = subparsers.add_parser(
        "auth",
        help="Authenticate with Mendeley",
        description="Initiate OAuth2 authentication with Mendeley API.",
    )
    auth_parser.set_defaults(func=cmd_auth)


def setup_query_parser(subparsers: Any) -> None:
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


def setup_test_parser(subparsers: Any) -> None:
    """Set up the test subcommand."""
    test_parser = subparsers.add_parser(
        "test",
        help="Test embedding and LLM connections",
        description="Test that the embedding model and LLM are working correctly.",
    )
    test_parser.add_argument(
        "--embedding-only",
        action="store_true",
        help="Only test the embedding model",
    )
    test_parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only test the LLM",
    )
    test_parser.set_defaults(func=cmd_test)


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
    agent = RAGAgent(config, auto_sync=True, save_responses=True)
    
    try:
        # Run the query (auto-syncs and adds documents automatically)
        print("Syncing references and preparing documents...")
        result = await agent.query(question)
        
        # Print with citations at the top
        print("\n" + "=" * 60)
        print("CITATIONS:")
        print("-" * 60)
        citations = result.get_citations()
        if citations:
            for i, citation in enumerate(citations, 1):
                print(f"  [{i}] {citation}")
        else:
            print("  No citations available")
        
        print("\n" + "=" * 60)
        print(f"Question: {result.question}")
        print("-" * 60)
        print(f"\nAnswer:\n{result.answer}")
        print("\n" + "=" * 60)
        print(f"Model: {result.model}")
        print(f"Response saved to: {config.responses_dir}/")
        
        return 0
    except Exception as e:
        print(f"Error during query: {e}")
        logger.exception("Query failed")
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    """Handle the test command."""
    config = Config.from_env()
    
    test_embedding = not args.llm_only
    test_llm = not args.embedding_only
    
    results = []
    
    if test_embedding:
        print("\n" + "=" * 60)
        print("TESTING EMBEDDING MODEL")
        print("=" * 60)
        embedding_result = _test_embedding(config)
        results.append(("Embedding", embedding_result))
    
    if test_llm:
        print("\n" + "=" * 60)
        print("TESTING LLM (Gemini)")
        print("=" * 60)
        llm_result = _test_llm(config)
        results.append(("LLM", llm_result))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    return 0 if all_passed else 1


def _test_embedding(config: Config) -> bool:
    """Test the embedding model connection.
    
    Returns:
        True if test passed, False otherwise.
    """
    embedding_config = config.embedding
    print(f"\nProvider: {embedding_config.provider}")
    print(f"Model: {embedding_config.model_name}")
    
    if embedding_config.provider == "ollama":
        print(f"Ollama URL: {embedding_config.ollama_url}")
        return _test_ollama_embedding(embedding_config)
    elif embedding_config.provider == "google":
        return _test_google_embedding(config)
    else:
        print(f"Unknown embedding provider: {embedding_config.provider}")
        return False


def _test_ollama_embedding(embedding_config: Any) -> bool:
    """Test Ollama embedding model."""
    import requests
    
    test_text = "This is a test sentence to generate embeddings."
    
    print(f"\nTesting embedding generation...")
    print(f"  Input text: \"{test_text}\"")
    
    try:
        # Test Ollama server connection
        url = f"{embedding_config.ollama_url}/api/embeddings"
        payload = {
            "model": embedding_config.model_name,
            "prompt": test_text,
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embedding = result.get("embedding", [])
        
        if embedding:
            print(f"  ✓ Embedding generated successfully!")
            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            return True
        else:
            print("  ✗ No embedding returned")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to Ollama at {embedding_config.ollama_url}")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  ✗ Model '{embedding_config.model_name}' not found")
            print(f"  Pull the model: ollama pull {embedding_config.model_name}")
        else:
            print(f"  ✗ HTTP error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def _test_google_embedding(config: Config) -> bool:
    """Test Google embedding model."""
    print("\n  Testing Google embedding...")
    
    if not config.gemini.api_key:
        print("  ✗ GEMINI_API_KEY not set")
        return False
    
    try:
        from google import genai
        
        client = genai.Client(api_key=config.gemini.api_key)
        
        test_text = "This is a test sentence to generate embeddings."
        print(f"  Input text: \"{test_text}\"")
        
        result = client.models.embed_content(
            model=config.embedding.model_name,
            contents=test_text,
        )
        
        embedding = result.embeddings[0].values if result.embeddings else []
        if embedding:
            print(f"  ✓ Embedding generated successfully!")
            print(f"  Embedding dimensions: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            return True
        else:
            print("  ✗ No embedding returned")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def _test_llm(config: Config) -> bool:
    """Test the LLM (Gemini) connection.
    
    Returns:
        True if test passed, False otherwise.
    """
    print(f"\nModel: {config.gemini.model_name}")
    
    if not config.gemini.api_key:
        print("  ✗ GEMINI_API_KEY not set")
        return False
    
    test_prompt = "Reply with exactly: 'Hello, BiblioRAG test successful!'"
    print(f"\nTesting LLM response generation...")
    print(f"  Prompt: \"{test_prompt}\"")
    
    try:
        from google import genai
        
        client = genai.Client(api_key=config.gemini.api_key)
        
        response = client.models.generate_content(
            model=config.gemini.model_name,
            contents=test_prompt,
        )
        
        if response and response.text:
            print(f"  ✓ LLM response received!")
            print(f"  Response: \"{response.text.strip()}\"")
            return True
        else:
            print("  ✗ No response generated")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


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
    setup_query_parser(subparsers)
    setup_test_parser(subparsers)
    
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
