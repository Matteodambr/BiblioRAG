# BiblioRAG

BiblioRAG is a Python wrapper to handle high-accuracy retrieval augmented generation (RAG) from your Mendeley library. This repo is built on top of the Mendeley API and on Paper-QA2, to automatically retrieve your reference lists, and perform RAG actions on them, such as question answering.

## Default Models

BiblioRAG uses the following models by default:

- **LLM (Language Model)**: `gemini-1.5-pro` - Google's Gemini Pro model for high-quality responses
- **Embedding Model**: `nomic-embed-text` - Local embeddings via Ollama (no API key required)

These can be customized via environment variables. See the [Configuration](#configuration) section for details.

## Features

- **Auto-Sync**: References are automatically synced from Mendeley before each query session
- **Smart Downloads**: Only download new or changed files to the `references/` folder
- **RAG-Powered Q&A**: Ask questions about your papers using Paper-QA2
- **Gemini Integration**: Uses Google Gemini Pro for high-quality responses
- **Local Embeddings**: Uses Ollama with nomic-embed-text for local embeddings (no OpenAI API needed)
- **Response Logging**: All interactions are automatically saved to the `responses/` folder
- **Proper Citations**: Citations display author names, year, and title (not just filenames)

## Installation

### Prerequisites

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) if you don't have conda installed.

### Quick Setup (Recommended)

Clone the repository and run the setup script. The setup script will automatically:
- Create the conda environment
- Install all dependencies
- Install Ollama (if not already installed)
- Pull the nomic-embed-text embedding model
- Start the Ollama service

```bash
git clone https://github.com/Matteodambr/BiblioRAG.git
cd BiblioRAG
```

**On Linux/macOS:**
```bash
./setup_env.sh
```

**On Windows:**
```cmd
setup_env.bat
```

The setup script will:
1. Create a conda environment named `bibliorag`
2. Install all dependencies
3. Install the BiblioRAG package
4. Create a `.env` file from the template

### Manual Installation

If you prefer to set up manually:

```bash
# Clone the repository
git clone https://github.com/Matteodambr/BiblioRAG.git
cd BiblioRAG

# Create and activate conda environment
conda env create -f environment.yml
conda activate bibliorag

# Install the package
pip install -e .

# Copy environment template
cp .env.example .env
```

### Activating the Environment

After initial setup, activate the environment before using BiblioRAG:

```bash
conda activate bibliorag
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Get your Mendeley API credentials:
   - Go to [Mendeley Developer Portal](https://dev.mendeley.com/)
   - Create a new application
   - Copy the Client ID and Client Secret to your `.env` file

3. Get your Gemini API key:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key
   - Add it to your `.env` file as `GEMINI_API_KEY`

4. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

## Usage

### 1. Test Your Setup

Before using BiblioRAG, verify that your embedding model and LLM are working:

```bash
bibliorag test
```

This will test:
- **Embedding model**: Connects to Ollama and generates a test embedding
- **LLM (Gemini)**: Sends a test prompt and verifies the response

You can also test them individually:
```bash
bibliorag test --embedding-only  # Test only the embedding model
bibliorag test --llm-only        # Test only the LLM
```

### 2. Authenticate with Mendeley

Authenticate with Mendeley to get access tokens:

```bash
bibliorag auth
```

Follow the prompts to complete OAuth2 authentication. Save the tokens in your `.env` file.

### 3. Query Your References

Ask questions about your papers (references are automatically synced):

```bash
bibliorag query "What are the main methods used for X?"
```

Each query will:
- Automatically sync new/updated references from Mendeley
- Display citations at the top of the response (with author, year, and title)
- Save the full interaction to the `responses/` folder

## How It Works

### Embedding and Indexing

BiblioRAG uses Paper-QA2 for document indexing and retrieval:

- **Local Embeddings**: By default, uses Ollama with `nomic-embed-text` for embeddings (no API key required)
- **Alternative**: Can also use Google embeddings by setting `EMBEDDING_PROVIDER=google` in `.env`
- **Incremental Processing**: Only new or updated documents are processed - existing embeddings are cached by Paper-QA2
- **Index Storage**: Paper-QA2 maintains its own internal index for efficient retrieval

### Sync State

BiblioRAG tracks which documents have been synced in `.bibliorag_state.json`:
- Document metadata (title, authors, year) for proper citation formatting
- Content hashes to detect changes
- File paths for downloaded PDFs

## Programmatic Usage

You can also use BiblioRAG as a library:

```python
import asyncio
from bibliorag import Config, RAGAgent

# Load configuration
config = Config.from_env()

# Query your documents (auto-syncs and saves responses by default)
async def main():
    agent = RAGAgent(config)  # auto_sync=True, save_responses=True by default
    
    result = await agent.query("What is the relationship between X and Y?")
    
    # Access citations (formatted with author, year, title)
    print("Citations:", result.get_citations())
    print("Model:", result.model)
    print("Answer:", result.answer)

asyncio.run(main())
```

### Response File Format

Each saved response follows this format:

```
================================================================================
BIBLIORAG QUERY RESPONSE
================================================================================

PROMPT:
----------------------------------------
<your question>

MODEL:
----------------------------------------
gemini-2.0-flash

CITATIONS:
----------------------------------------
  [1] Smith et al. (2023) - Title of First Paper
  [2] Jones & Brown (2022) - Title of Second Paper

RESPONSE:
----------------------------------------
<the answer>

================================================================================
Timestamp: 2024-01-15T10:30:00
================================================================================
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MENDELEY_CLIENT_ID` | Your Mendeley application Client ID |
| `MENDELEY_CLIENT_SECRET` | Your Mendeley application Client Secret |
| `MENDELEY_ACCESS_TOKEN` | OAuth2 access token (obtained via `auth` command) |
| `MENDELEY_REFRESH_TOKEN` | OAuth2 refresh token (obtained via `auth` command) |
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `EMBEDDING_PROVIDER` | Embedding provider: `ollama` (default) or `google` |
| `EMBEDDING_MODEL` | Embedding model: `nomic-embed-text` (default for Ollama) or `models/embedding-001` (for Google) |
| `OLLAMA_URL` | Ollama server URL (default: `http://localhost:11434`) |

## Project Structure

```
BiblioRAG/
├── src/
│   └── bibliorag/
│       ├── __init__.py      # Package initialization
│       ├── cli.py           # Command-line interface
│       ├── config.py        # Configuration management
│       ├── mendeley_client.py  # Mendeley API client
│       └── rag_agent.py     # RAG agent wrapper
├── references/              # Downloaded PDF files (gitignored)
├── responses/               # Saved query responses (gitignored)
├── environment.yml          # Conda environment specification
├── setup_env.sh            # Setup script for Linux/macOS
├── setup_env.bat           # Setup script for Windows
├── .env.example            # Example environment configuration
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
