#!/bin/bash
# Startup script for fractional-calculus-library project
# This script automatically activates the fracnn conda environment

echo "🚀 Starting fractional-calculus-library environment..."

# Initialize conda if not already done
if ! command -v conda &> /dev/null; then
    echo "📦 Initializing conda..."
    source /c/Users/davia/miniconda3/etc/profile.d/conda.sh
fi

# Check if fracnn environment exists
if conda env list | grep -q "fracnn"; then
    echo "✅ Found fracnn environment, activating..."
    conda activate fracnn
    echo "🎉 Environment activated! Current environment: $CONDA_DEFAULT_ENV"
    
    # Set project-specific environment variables
    export PYTHONPATH="${PWD}:${PYTHONPATH}"
    export HPFRACC_PROJECT_ROOT="${PWD}"
    
    echo ""
    echo "📊 Environment Information:"
    echo "   Environment: $CONDA_DEFAULT_ENV"
    echo "   Python version: $(python --version)"
    echo "   Working directory: $(pwd)"
    echo "   PYTHONPATH: ${PYTHONPATH}"
    echo ""
    echo "🚀 Quick Commands:"
    echo "   To run tests: python -m pytest tests/"
    echo "   To run examples: python examples/basic_usage/getting_started.py"
    echo "   To check environment: conda info --envs"
    echo "   To deactivate: conda deactivate"
else
    echo "❌ fracnn environment not found!"
    echo "Available environments:"
    conda env list
    echo ""
    echo "To create the environment, run:"
    echo "  conda env create -f config/environment.yml"
fi
