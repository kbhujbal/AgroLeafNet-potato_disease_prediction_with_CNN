#!/bin/bash
# Training script for AgroLeafNet
# Uses the same Python environment as Jupyter

echo "Starting AgroLeafNet Training"
echo "=============================="
echo ""

# Use the Homebrew Python (same as Jupyter)
PYTHON_PATH="/opt/homebrew/Cellar/jupyterlab/4.3.5_1/libexec/bin/python"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python not found at $PYTHON_PATH"
    echo "Using system python instead..."
    PYTHON_PATH="python3"
fi

# Run training
cd src
$PYTHON_PATH train.py
