@echo off
echo Activating fracnn conda environment...
call conda activate fracnn
echo.
echo Environment activated! You can now:
echo - Run tests: pytest
echo - Run benchmarks: python benchmarks/comprehensive_performance_benchmark.py
echo - Build package: python -m build
echo - Upload to PyPI: twine upload dist/*
echo.
cmd /k
