#!/bin/bash
# Setup script for BiblioRAG on Linux/macOS
# This script creates and activates the conda environment

set -e

ENV_NAME="bibliorag"

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
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Mendeley and Gemini API credentials"
echo "  2. Run 'bibliorag auth' to authenticate with Mendeley"
echo "  3. Run 'bibliorag query \"your question\"' to query your references"
echo ""
