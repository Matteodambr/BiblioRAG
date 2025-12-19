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
    test_parser.add_argument(
        "--enrichment-only",
        action="store_true",
        help="Only test the enrichment model",
    )
    test_parser.set_defaults(func=cmd_test)


def setup_models_parser(subparsers: Any) -> None:
    """Set up the models subcommand."""
    models_parser = subparsers.add_parser(
        "models",
        help="List available Gemini models",
        description="List all available Gemini models that support text generation.",
    )
    models_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all models including embeddings and specialized models",
    )
    models_parser.set_defaults(func=cmd_models)


def setup_clear_cache_parser(subparsers: Any) -> None:
    """Set up the clear-cache subcommand."""
    clear_cache_parser = subparsers.add_parser(
        "clear-cache",
        help="Clear the document index cache",
        description="Clear the cached document index, forcing re-indexing on next query.",
    )
    clear_cache_parser.set_defaults(func=cmd_clear_cache)


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
        print("\n✓ Authentication successful!")
        print("✓ Tokens saved automatically to .bibliorag_state.json")
        print("\nYou can now use BiblioRAG commands:")
        print("  bibliorag query \"Your question here\"")
        return 0
    except Exception as e:
        print(f"Error during authentication: {e}")
        return 1


def cmd_query(args: argparse.Namespace) -> int:
    """Handle the query command."""
    config = Config.from_env()
    
    # Validate LLM configuration
    if config.llm.provider == "gemini" and not config.llm.api_key:
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
    
    # Determine which tests to run
    if args.embedding_only or args.llm_only or args.enrichment_only:
        test_embedding = args.embedding_only
        test_llm = args.llm_only
        test_enrichment = args.enrichment_only
    else:
        # Run all tests by default
        test_embedding = True
        test_llm = True
        test_enrichment = True
    
    results = []
    
    if test_embedding:
        print("\n" + "=" * 60)
        print("TESTING EMBEDDING MODEL")
        print("=" * 60)
        embedding_result = _test_embedding(config)
        results.append(("Embedding", embedding_result))
    
    if test_enrichment:
        print("\n" + "=" * 60)
        print("TESTING ENRICHMENT MODEL (for indexing)")
        print("=" * 60)
        enrichment_result = _test_enrichment(config)
        results.append(("Enrichment", enrichment_result))
    
    if test_llm:
        print("\n" + "=" * 60)
        print("TESTING LLM (for queries)")
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
    
    if not config.llm.api_key:
        print("  ✗ GEMINI_API_KEY not set")
        return False
    
    try:
        from google import genai
        
        client = genai.Client(api_key=config.llm.api_key)
        
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


def _check_model_availability(config: Config) -> tuple[bool, list[str]]:
    """Check if the selected model is available and list available models.
    
    Returns:
        Tuple of (is_available, list_of_available_models)
    """
    try:
        from google import genai
        import requests
        
        # Use direct API call to list models
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config.llm.api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return False, []
        
        data = response.json()
        models = data.get('models', [])
        
        # Get models that support generateContent
        available_models = []
        for model in models:
            methods = model.get('supportedGenerationMethods', [])
            if 'generateContent' in methods:
                model_name = model['name'].replace('models/', '')
                available_models.append(model_name)
        
        # Check if configured model is in the list
        is_available = config.llm.model_name in available_models
        
        return is_available, available_models
        
    except Exception as e:
        logger.debug(f"Error checking model availability: {e}")
        return False, []


def _test_llm(config: Config) -> bool:
    """Test the LLM connection.
    
    Returns:
        True if test passed, False otherwise.
    """
    if config.llm.provider == "ollama":
        model_name = f"ollama/{config.llm.model_name}"
        return _test_ollama_llm(model_name, config)
    elif config.llm.provider == "gemini":
        return _test_gemini_llm(config)
    else:
        print(f"Unknown LLM provider: {config.llm.provider}")
        return False


def _test_gemini_llm(config: Config) -> bool:
    """Test Gemini LLM connection.
    
    Returns:
        True if test passed, False otherwise.
    """
    print(f"\nConfigured Model: {config.llm.model_name}")
    
    if not config.llm.api_key:
        print("  ✗ GEMINI_API_KEY not set")
        return False
    
    # Check model availability
    print("\nChecking model availability...")
    is_available, available_models = _check_model_availability(config)
    
    if not is_available:
        print(f"  ✗ Model '{config.llm.model_name}' is not available")
        if available_models:
            print("\n  Available models that support text generation:")
            for i, model in enumerate(sorted(available_models)[:10], 1):
                print(f"    {i}. {model}")
            if len(available_models) > 10:
                print(f"    ... and {len(available_models) - 10} more")
            print("\n  Update GEMINI_MODEL in your .env file to use one of these models.")
        return False
    else:
        print(f"  ✓ Model is available ({len(available_models)} models total)")
    
    test_prompt = "Reply with exactly: 'Hello, BiblioRAG test successful!'"
    print(f"\nTesting LLM response generation...")
    print(f"  Prompt: \"{test_prompt}\"")
    
    try:
        from google import genai
        
        client = genai.Client(api_key=config.llm.api_key)
        
        response = client.models.generate_content(
            model=config.llm.model_name,
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

def _test_enrichment(config: Config) -> bool:
    """Test the enrichment model connection.
    
    Returns:
        True if test passed, False otherwise.
    """
    # Determine the enrichment model
    if config.llm.enrichment_model:
        enrichment_model = config.llm.enrichment_model
    else:
        # Use the main LLM model
        if config.llm.provider == "gemini":
            enrichment_model = f"gemini/{config.llm.model_name}"
        else:
            enrichment_model = f"ollama/{config.llm.model_name}"
    
    print(f"\nConfigured Model: {enrichment_model}")
    
    # Check if it's same as main LLM
    if not config.llm.enrichment_model:
        print("  ⓘ Using same model as main LLM (no separate enrichment model configured)")
        print("  ⓘ To use a local model for indexing, set ENRICHMENT_MODEL in .env")
        print("  ⓘ Example: ENRICHMENT_MODEL=ollama/llama3.2:1b")
        return True  # If not configured, it will use the main LLM which was already tested
    
    # Determine if it's an Ollama or Gemini model
    if enrichment_model.startswith("ollama/"):
        return _test_ollama_llm(enrichment_model, config)
    elif enrichment_model.startswith("gemini/"):
        # Strip gemini/ prefix for testing
        model_name = enrichment_model.replace("gemini/", "")
        temp_config = Config.from_env()
        temp_config.llm.model_name = model_name
        return _test_gemini_llm(temp_config)
    else:
        print(f"  ⚠ Unknown model format: {enrichment_model}")
        print("  Expected format: 'ollama/model-name' or 'gemini/model-name'")
        return False


def _test_ollama_llm(model_name: str, config: Config) -> bool:
    """Test Ollama LLM model.
    
    Args:
        model_name: Full model name (e.g., 'ollama/llama3.2:1b')
        config: Configuration object
        
    Returns:
        True if test passed, False otherwise.
    """
    import requests
    
    # Extract model name without ollama/ prefix
    ollama_model = model_name.replace("ollama/", "")
    ollama_url = config.embedding.ollama_url
    
    print(f"  Ollama URL: {ollama_url}")
    print(f"  Model: {ollama_model}")
    
    test_prompt = "Reply with exactly: 'Hello, BiblioRAG test successful!'"
    print(f"\nTesting LLM response generation...")
    print(f"  Prompt: \"{test_prompt}\"")
    
    try:
        # Test Ollama server connection
        url = f"{ollama_url}/api/generate"
        payload = {
            "model": ollama_model,
            "prompt": test_prompt,
            "stream": False,
        }
        
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get("response", "")
        
        if response_text:
            print(f"  ✓ LLM response received!")
            print(f"  Response: \"{response_text.strip()}\"")
            return True
        else:
            print("  ✗ No response text returned")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Cannot connect to Ollama at {ollama_url}")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  ✗ Model '{ollama_model}' not found")
            print(f"  Pull the model: ollama pull {ollama_model}")
        else:
            print(f"  ✗ HTTP error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def cmd_models(args: argparse.Namespace) -> int:
    """Handle the models command."""
    config = Config.from_env()
    
    if config.llm.provider != "gemini":
        print(f"This command only works with Gemini provider.")
        print(f"Current provider: {config.llm.provider}")
        return 1
    
    if not config.llm.api_key:
        print("Error: Gemini API key not configured.")
        print("Please set GEMINI_API_KEY environment variable.")
        return 1
    
    try:
        import requests
        
        print("Fetching available Gemini models...\n")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={config.gemini.api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: Failed to fetch models (HTTP {response.status_code})")
            return 1
        
        data = response.json()
        models = data.get('models', [])
        
        # Filter by text generation unless --all is specified
        if not args.all:
            models = [
                m for m in models
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
        
        if not models:
            print("No models found.")
            return 0
        
        # Display current configuration
        print("=" * 70)
        print(f"Currently configured model: {config.gemini.model_name}")
        print("=" * 70)
        print()
        
        # Group models by category
        print(f"Available models ({len(models)} total):\n")
        
        for model in sorted(models, key=lambda m: m.get('name', '')):
            name = model.get('name', '').replace('models/', '')
            display_name = model.get('displayName', name)
            description = model.get('description', 'N/A')
            methods = model.get('supportedGenerationMethods', [])
            
            # Highlight if this is the configured model
            marker = " ← CONFIGURED" if name == config.llm.model_name else ""
            
            print(f"• {name}{marker}")
            print(f"  Display: {display_name}")
            if description and description != 'N/A' and len(description) < 100:
                print(f"  Description: {description}")
            if args.all:
                print(f"  Methods: {', '.join(methods)}")
            print()
        
        if not args.all:
            print("\nNote: Use --all to see all models including embeddings and specialized models.")
        
        print("\nTo change the model, update GEMINI_MODEL in your .env file.")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Failed to list models")
        return 1


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """Handle the clear-cache command."""
    config = Config.from_env()
    agent = RAGAgent(config, auto_sync=False, save_responses=False)
    
    print("Clearing document index cache...")
    agent.clear_cache()
    
    return 0


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
    setup_models_parser(subparsers)
    setup_clear_cache_parser(subparsers)
    
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
