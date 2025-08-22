# .bashrc for hpfracc project
# This file automatically activates the fracnn conda environment

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# Initialize conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# Auto-activate fracnn environment if not already active
if [ "$CONDA_DEFAULT_ENV" != "fracnn" ]; then
    echo "🐍 Auto-activating fracnn conda environment..."
    conda activate fracnn
    echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
fi

# Set up project-specific aliases
alias test-package="python test_package_functionality.py"
alias run-tests="pytest tests/ -v"
alias build-package="python -m build"
alias upload-pypi="twine upload dist/*"

# Show helpful information
echo "🚀 hpfracc development environment ready!"
echo "📁 Current directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🌍 Conda env: $CONDA_DEFAULT_ENV"
echo ""
echo "💡 Useful commands:"
echo "  test-package    - Run package functionality tests"
echo "  run-tests       - Run all tests with pytest"
echo "  build-package   - Build the package"
echo "  upload-pypi     - Upload to PyPI"
echo ""
