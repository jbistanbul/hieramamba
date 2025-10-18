#!/bin/bash

# HieraMamba Installation Script
echo "Installing HieraMamba..."

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: This script should be run from the root of the HieraMamba repository."
    echo "Please clone the repository first with: git clone --recursive https://github.com/jbistanbul/hieramamba.git"
    exit 1
fi

# Initialize submodules if not already done
echo "Initializing git submodules..."
git submodule update --init --recursive

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Use python3 if python is not available
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Install requirements
echo "Installing Python dependencies..."
echo "Note: Please install Mamba SSM following the official instructions: https://github.com/state-spaces/mamba"
$PYTHON_CMD -m pip install -r requirements.txt

echo "Installation complete!"
echo ""
echo "To verify the installation, you can run:"
echo "  $PYTHON_CMD -c \"import torch; from hydra.modules.hydra import Hydra; from mamba_ssm import Mamba2; print('Installation successful!')\""
