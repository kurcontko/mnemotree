#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

VENV_PATH=${VENV_PATH:-.venv}
PORT_STREAMLIT=${STREAMLIT_PORT:-7860}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it from https://github.com/astral-sh/uv"
  exit 1
fi

# Create Python virtual environment
echo "Creating uv virtual environment \"$VENV_PATH\" in root directory..."
uv venv "$VENV_PATH"

# Install dependencies with UI extras
echo "Installing project dependencies with UI extras..."
uv pip install -e ".[ui]" --python "$VENV_PATH/bin/python"

# Run the Streamlit app
uv run --python "$VENV_PATH/bin/python" streamlit run examples/memory_chat/app.py --server.port "$PORT_STREAMLIT"
