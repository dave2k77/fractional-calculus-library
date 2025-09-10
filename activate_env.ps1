# PowerShell script to activate conda environment
Write-Host "Activating fracnn conda environment..." -ForegroundColor Green
conda activate fracnn
Write-Host "Environment activated! Current environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor Green
