#!/bin/bash
# Git Bash script to activate the fracnn conda environment
# Usage: source activate_env.sh

echo "🐍 Activating fracnn conda environment..."

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate fracnn

echo "✅ Environment activated!"
echo "🌍 Current environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python version: $(python --version)"
echo ""
echo "🚀 You can now:"
echo "  - Run tests: pytest"
echo "  - Run benchmarks: python benchmarks/comprehensive_performance_benchmark.py"
echo "  - Build package: python -m build"
echo "  - Upload to PyPI: twine upload dist/*"
echo "  - Test package: python test_package_functionality.py"
echo ""

# Keep the shell open
exec bash
