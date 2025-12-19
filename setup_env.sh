#!/bin/bash
# Setup script for BiblioRAG on Linux/macOS
# This script creates and activates the conda environment and sets up Ollama

set -e

ENV_NAME="bibliorag"
OLLAMA_MODEL="nomic-embed-text"

echo "=========================================="
echo "BiblioRAG Environment Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed."
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Initialize conda for the current shell if needed
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating environment..."
        conda env update -f environment.yml --prune
    fi
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating environment..."
conda activate ${ENV_NAME}

# Install the package in development mode
echo "Installing BiblioRAG package..."
pip install -e .

# Copy .env.example if .env doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env file from .env.example"
        echo "Please edit .env with your API credentials."
    fi
fi

echo ""
echo "=========================================="
echo "Setting up Ollama for local embeddings"
echo "=========================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Installing..."
    
    # Detect OS and install Ollama
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detected Linux. Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS."
        if command -v brew &> /dev/null; then
            echo "Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "Please install Ollama manually from https://ollama.ai/"
            echo "Or install Homebrew first: https://brew.sh/"
        fi
    else
        echo "Please install Ollama manually from https://ollama.ai/"
    fi
fi

# Check if Ollama is now available
if command -v ollama &> /dev/null; then
    echo "Ollama is installed."
    
    # Check if Ollama service is running
    if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo "Starting Ollama service..."
        # Start Ollama in the background
        ollama serve &> /dev/null &
        OLLAMA_PID=$!
        echo "Ollama started (PID: $OLLAMA_PID)"
        # Wait for Ollama to start
        echo "Waiting for Ollama to be ready..."
        sleep 3
        for i in {1..10}; do
            if curl -s http://localhost:11434/api/tags &> /dev/null; then
                echo "Ollama is ready!"
                break
            fi
            sleep 1
        done
    else
        echo "Ollama service is already running."
    fi
    
    # Check if model is already pulled
    if ollama list | grep -q "${OLLAMA_MODEL}"; then
        echo "Model '${OLLAMA_MODEL}' is already downloaded."
    else
        echo "Pulling embedding model '${OLLAMA_MODEL}'..."
        ollama pull ${OLLAMA_MODEL}
        echo "Model downloaded successfully!"
    fi
else
    echo "WARNING: Ollama installation may have failed."
    echo "Please install Ollama manually from https://ollama.ai/"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To start Ollama (if not running), run:"
echo "  ollama serve"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Mendeley and Gemini API credentials"
echo "  2. Run 'bibliorag test' to verify embedding and LLM setup"
echo "  3. Run 'bibliorag auth' to authenticate with Mendeley"
echo "  4. Run 'bibliorag query \"your question\"' to query your references"
echo ""
