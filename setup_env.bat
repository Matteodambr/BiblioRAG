@echo off
REM Setup script for BiblioRAG on Windows
REM This script creates and activates the conda environment and sets up Ollama

setlocal enabledelayedexpansion

set ENV_NAME=bibliorag
set OLLAMA_MODEL=nomic-embed-text

echo ==========================================
echo BiblioRAG Environment Setup
echo ==========================================

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda is not installed.
    echo Please install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM Initialize conda for cmd
call conda activate base

REM Check if environment already exists
conda env list | findstr /B /C:"%ENV_NAME% " >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Environment '%ENV_NAME%' already exists.
    set /p UPDATE="Do you want to update it? (y/n) "
    if /i "!UPDATE!"=="y" (
        echo Updating environment...
        call conda env update -f environment.yml --prune
    )
) else (
    echo Creating conda environment '%ENV_NAME%'...
    call conda env create -f environment.yml
)

REM Activate the environment
echo Activating environment...
call conda activate %ENV_NAME%

REM Install the package in development mode
echo Installing BiblioRAG package...
pip install -e .

REM Copy .env.example if .env doesn't exist
if not exist .env (
    if exist .env.example (
        copy .env.example .env
        echo Created .env file from .env.example
        echo Please edit .env with your API credentials.
    )
)

echo.
echo ==========================================
echo Setting up Ollama for local embeddings
echo ==========================================

REM Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Ollama is not installed.
    echo.
    echo Please download and install Ollama from:
    echo   https://ollama.ai/download/windows
    echo.
    echo After installation, run this script again or manually run:
    echo   ollama pull %OLLAMA_MODEL%
    echo.
    set /p INSTALL_OLLAMA="Would you like to open the Ollama download page? (y/n) "
    if /i "!INSTALL_OLLAMA!"=="y" (
        start https://ollama.ai/download/windows
    )
    goto :skip_ollama
)

echo Ollama is installed.

REM Check if Ollama service is running by attempting to connect
curl -s http://localhost:11434/api/tags >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Starting Ollama service...
    start /B ollama serve
    echo Waiting for Ollama to be ready...
    timeout /t 5 /nobreak >nul
)

REM Check if model is already pulled
ollama list | findstr /C:"%OLLAMA_MODEL%" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Model '%OLLAMA_MODEL%' is already downloaded.
) else (
    echo Pulling embedding model '%OLLAMA_MODEL%'...
    ollama pull %OLLAMA_MODEL%
    echo Model downloaded successfully!
)

:skip_ollama

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the environment in the future, run:
echo   conda activate %ENV_NAME%
echo.
echo To start Ollama (if not running), run:
echo   ollama serve
echo.
echo Next steps:
echo   1. Edit .env with your Mendeley and Gemini API credentials
echo   2. Run 'bibliorag test' to verify embedding and LLM setup
echo   3. Run 'bibliorag auth' to authenticate with Mendeley
echo   4. Run 'bibliorag query "your question"' to query your references
echo.

endlocal
