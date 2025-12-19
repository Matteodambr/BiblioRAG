# BiblioRAG

BiblioRAG is a Python wrapper to handle high-accuracy retrieval augmented generation (RAG) from your Mendeley library. This repo is built on top of the Mendeley API and on Paper-QA2, to automatically retrieve your reference lists, and perform RAG actions on them, such as question answering, summarization, and contradiction detection.

## Features

- **Mendeley Sync**: Automatically sync your Mendeley library, detecting new and updated references
- **Smart Downloads**: Only download new or changed files to the `references/` folder
- **RAG-Powered Q&A**: Ask questions about your papers using Paper-QA2
- **Summarization**: Generate summaries across your document collection
- **Contradiction Detection**: Find conflicting findings across papers
- **Gemini Integration**: Uses Google Gemini Pro for high-quality responses

## Installation

```bash
# Clone the repository
git clone https://github.com/Matteodambr/BiblioRAG.git
cd BiblioRAG

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
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

## Usage

### 1. Authenticate with Mendeley

First, authenticate with Mendeley to get access tokens:

```bash
bibliorag auth
```

Follow the prompts to complete OAuth2 authentication. Save the tokens in your `.env` file.

### 2. Sync References

Download new or updated references from your Mendeley library:

```bash
bibliorag sync
```

This will:
- Fetch your document list from Mendeley
- Detect new or updated documents
- Download associated PDF files to the `references/` folder
- Track sync state to avoid re-downloading unchanged files

### 3. Query Your References

Ask questions about your papers:

```bash
bibliorag query "What are the main methods used for X?"
```

### 4. Generate Summaries

Summarize the key findings across your documents:

```bash
bibliorag summarize

# With a specific focus
bibliorag summarize --focus "What are the main contributions in the field of X?"
```

### 5. Find Contradictions

Identify conflicting findings across your papers:

```bash
bibliorag contradictions
```

## Programmatic Usage

You can also use BiblioRAG as a library:

```python
import asyncio
from bibliorag import Config, MendeleyClient, RAGAgent

# Load configuration
config = Config.from_env()

# Sync Mendeley references
client = MendeleyClient(config)
updated_docs, downloaded_files = client.sync_references()

# Query your documents
async def main():
    agent = RAGAgent(config)
    await agent.add_documents()
    
    result = await agent.query("What is the relationship between X and Y?")
    print(result.answer)

asyncio.run(main())
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MENDELEY_CLIENT_ID` | Your Mendeley application Client ID |
| `MENDELEY_CLIENT_SECRET` | Your Mendeley application Client Secret |
| `MENDELEY_ACCESS_TOKEN` | OAuth2 access token (obtained via `auth` command) |
| `MENDELEY_REFRESH_TOKEN` | OAuth2 refresh token (obtained via `auth` command) |
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `EMBEDDING_MODEL` | (Optional) Embedding model to use (default: `models/embedding-001`) |

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
├── .env.example            # Example environment configuration
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
