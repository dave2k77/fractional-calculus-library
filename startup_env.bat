@echo off
echo 🚀 Starting fractional-calculus-library environment...

REM Initialize conda
call C:\Users\davia\miniconda3\Scripts\activate.bat

REM Activate fracnn environment
echo 📦 Activating fracnn environment...
call conda activate fracnn

if %ERRORLEVEL% EQU 0 (
    echo ✅ Environment activated! Current environment: %CONDA_DEFAULT_ENV%
    
    REM Set project-specific environment variables
    set PYTHONPATH=%CD%;%PYTHONPATH%
    set HPFRACC_PROJECT_ROOT=%CD%
    
    echo.
    echo 📊 Environment Information:
    echo    Environment: %CONDA_DEFAULT_ENV%
    python --version
    echo    Working directory: %CD%
    echo    PYTHONPATH: %PYTHONPATH%
    echo.
    echo 🚀 Quick Commands:
    echo    To run tests: python -m pytest tests/
    echo    To run examples: python examples/basic_usage/getting_started.py
    echo    To check environment: conda info --envs
    echo    To deactivate: conda deactivate
) else (
    echo ❌ Failed to activate fracnn environment!
    echo Available environments:
    conda env list
    echo.
    echo To create the environment, run:
    echo   conda env create -f environment.yml
)

cmd /k
