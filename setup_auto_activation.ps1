# PowerShell script to set up automatic conda environment activation
# This script should be run once to configure automatic activation

Write-Host "Setting up automatic conda environment activation for HPFRACC project..." -ForegroundColor Green

# Create a .vscode/settings.json file to configure the terminal
$vscodeDir = ".vscode"
if (!(Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir
    Write-Host "Created .vscode directory" -ForegroundColor Yellow
}

# Create settings.json for VS Code/Cursor
$settings = @{
    "terminal.integrated.defaultProfile.windows" = "Git Bash"
    "terminal.integrated.profiles.windows" = @{
        "Git Bash" = @{
            "path" = "C:\Program Files\Git\bin\bash.exe"
            "args" = @("-l")
        }
        "PowerShell" = @{
            "path" = "powershell.exe"
            "args" = @("-NoExit", "-Command", "conda activate fracnn")
        }
    }
    "python.defaultInterpreterPath" = "C:\Users\davia\miniconda3\envs\fracnn\python.exe"
    "python.terminal.activateEnvironment" = $true
} | ConvertTo-Json -Depth 10

$settings | Out-File -FilePath ".vscode/settings.json" -Encoding UTF8
Write-Host "Created .vscode/settings.json with conda environment configuration" -ForegroundColor Yellow

# Create a .envrc file for direnv (if available)
$envrc = @"
#!/bin/bash
# Automatically activate conda environment
conda activate fracnn
"@

$envrc | Out-File -FilePath ".envrc" -Encoding UTF8
Write-Host "Created .envrc file for direnv support" -ForegroundColor Yellow

# Create a conda environment activation script
$activateScript = @"
@echo off
echo Activating fracnn conda environment...
call conda activate fracnn
echo Environment activated! Current environment: %CONDA_DEFAULT_ENV%
"@

$activateScript | Out-File -FilePath "activate_env.bat" -Encoding ASCII
Write-Host "Created activate_env.bat for manual activation" -ForegroundColor Yellow

# Create a PowerShell activation script
$psActivateScript = @"
# PowerShell script to activate conda environment
Write-Host "Activating fracnn conda environment..." -ForegroundColor Green
conda activate fracnn
Write-Host "Environment activated! Current environment: `$env:CONDA_DEFAULT_ENV" -ForegroundColor Green
"@

$psActivateScript | Out-File -FilePath "activate_env.ps1" -Encoding UTF8
Write-Host "Created activate_env.ps1 for PowerShell activation" -ForegroundColor Yellow

Write-Host "`nSetup complete! The following files have been created:" -ForegroundColor Green
Write-Host "- .vscode/settings.json (VS Code/Cursor terminal configuration)" -ForegroundColor White
Write-Host "- .envrc (direnv support)" -ForegroundColor White
Write-Host "- activate_env.bat (Windows batch activation)" -ForegroundColor White
Write-Host "- activate_env.ps1 (PowerShell activation)" -ForegroundColor White

Write-Host "`nTo use:" -ForegroundColor Cyan
Write-Host "1. Restart VS Code/Cursor to pick up the new terminal settings" -ForegroundColor White
Write-Host "2. Or manually run: .\activate_env.ps1" -ForegroundColor White
Write-Host "3. Or manually run: .\activate_env.bat" -ForegroundColor White

Write-Host "`nThe fracnn conda environment will now be automatically activated when opening a terminal in this directory." -ForegroundColor Green
