#!/usr/bin/env bash
set -euo pipefail

# Run minimal probabilistic fractional training example and verify output
# Set JAX to CPU-only mode for smoke tests to avoid GPU initialization issues
export JAX_PLATFORM_NAME=cpu

PYTHONPATH="$(pwd)" python examples/ml_examples/minimal_probabilistic_fractional_training.py 2>&1 | tee smoke_output.txt

# Check if training completed successfully
if grep -q "Training completed." smoke_output.txt; then
    echo "Smoke test passed."
    exit 0
else
    echo "Smoke test failed: 'Training completed.' not found in output"
    echo "Last 20 lines of output:"
    tail -20 smoke_output.txt
    exit 1
fi
