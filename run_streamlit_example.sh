#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

VENV_PATH=${VENV_PATH:-.venv}
PORT_STREAMLIT=${STREAMLIT_PORT:-7860}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Virtual environment not found at $VENV_PATH"
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Install dependencies with UI extras if not already installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing project dependencies with UI extras..."
    pip install -e ".[ui]"
fi

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY is not set. Please set it to use the app."
    echo "Example: export OPENAI_API_KEY='your-api-key'"
fi

# Run the Streamlit app from the examples/memory_chat directory
echo "Starting Streamlit app on port $PORT_STREAMLIT..."
cd examples/memory_chat
streamlit run app.py --server.port "$PORT_STREAMLIT"
