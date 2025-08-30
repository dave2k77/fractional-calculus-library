#!/bin/bash
# Activation script for hpfracc-env
# Usage: source activate_env.sh

echo "Activating hpfracc-env environment..."
conda activate hpfracc-env

if [ $? -eq 0 ]; then
    echo "✅ Environment activated successfully!"
    echo "Python version: $(python --version)"
    echo "Environment: $(conda info --envs | grep '*' | awk '{print $1}')"
    echo ""
    echo "To deactivate, run: conda deactivate"
    echo "To run tests: python -m pytest tests/"
    echo "To run examples: python examples/basic_usage/getting_started.py"
else
    echo "❌ Failed to activate environment"
    echo "Make sure conda is available and hpfracc-env exists"
fi
