#!/usr/bin/env bash
set -euo pipefail

# Run minimal probabilistic fractional training example and verify output
PYTHONPATH="$(pwd)" python examples/minimal_probabilistic_fractional_training.py | tee smoke_output.txt

grep -q "Training completed." smoke_output.txt

echo "Smoke test passed."
