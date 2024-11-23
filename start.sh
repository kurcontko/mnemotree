#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

VENV_PATH=${VENV_PATH:-.venv}
PORT_STREAMLIT=${STREAMLIT_PORT:-7860}

# Create Python virtual environment
echo "Creating Python virtual environment \"$VENV_PATH\" in root directory..."
python3 -m venv "$VENV_PATH"

# Activate the virtual environment
echo "Activating the virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from "requirements.txt" into the virtual environment
echo "Installing dependencies from \"requirements.txt\" into virtual environment..."
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py --server.port "$PORT_STREAMLIT"