#!/bin/bash
# Quick training script - uses Jupyter's Python environment

cd "$(dirname "$0")/src"
/opt/homebrew/Cellar/jupyterlab/4.3.5_1/libexec/bin/python train.py
