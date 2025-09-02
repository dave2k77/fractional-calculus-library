#!/bin/bash
# Activation script for hpfracc environment
# Usage: source activate_env.sh

echo "🔍 Looking for available conda environments..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not available. Please install conda first."
    exit 1
fi

# Try to activate hpfracc-env first, then fracnn if that fails
if conda env list | grep -q "hpfracc-env"; then
    echo "📦 Found hpfracc-env environment, activating..."
    conda activate hpfracc-env
    if [ $? -eq 0 ]; then
        echo "✅ hpfracc-env activated successfully!"
        ENV_NAME="hpfracc-env"
    else
        echo "❌ Failed to activate hpfracc-env"
        exit 1
    fi
elif conda env list | grep -q "fracnn"; then
    echo "📦 Found fracnn environment, activating..."
    conda activate fracnn
    if [ $? -eq 0 ]; then
        echo "✅ fracnn activated successfully!"
        ENV_NAME="fracnn"
    else
        echo "❌ Failed to activate fracnn"
        exit 1
    fi
else
    echo "❌ No suitable conda environment found."
    echo ""
    echo "Available options:"
    echo "1. Create hpfracc-env: conda create -n hpfracc-env python=3.11"
    echo "2. Create from environment.yml: conda env create -f environment.yml"
    echo ""
    echo "After creating an environment, run this script again."
    exit 1
fi

# Set project-specific environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export HPFRACC_PROJECT_ROOT="${PWD}"

echo ""
echo "📊 Environment Information:"
echo "   Environment: $ENV_NAME"
echo "   Python version: $(python --version)"
echo "   Working directory: $(pwd)"
echo "   PYTHONPATH: ${PYTHONPATH}"
echo ""
echo "🚀 Quick Commands:"
echo "   To deactivate: conda deactivate"
echo "   To run tests: python -m pytest tests/"
echo "   To run examples: python examples/basic_usage/getting_started.py"
echo "   To check environment: conda info --envs"
