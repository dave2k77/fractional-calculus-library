# PyPI Setup Guide for HPFRACC

## Overview
This guide explains how to set up PyPI authentication for uploading the `hpfracc` package.

## Prerequisites
- PyPI account with API token access
- `twine` package installed (`pip install twine`)

## Setup Steps

### 1. Create PyPI API Token
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Navigate to "API tokens"
3. Create a new token with scope: `hpfracc`
4. Copy the token (it starts with `pypi-`)

### 2. Configure Authentication
Create a `.pypirc` file in your home directory (`~/.pypirc`):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = YOUR_PYPI_API_TOKEN_HERE
repository = https://upload.pypi.org/legacy/

[testpypi]
username = __token__
password = YOUR_TESTPYPI_API_TOKEN_HERE
repository = https://test.pypi.org/legacy/
```

**Important**: Replace `YOUR_PYPI_API_TOKEN_HERE` with your actual token.

### 3. Build and Upload
```bash
# Build the package
python -m build

# Upload to PyPI
twine upload dist/*

# Upload to TestPyPI (optional)
twine upload --repository testpypi dist/*
```

## Security Notes
- **Never commit `.pypirc` to git** - it contains sensitive API tokens
- The `.pypirc` file is already in `.gitignore`
- Use `.pypirc.template` as a reference for the file structure

## Troubleshooting
- If you get authentication errors, verify your API token is correct
- Ensure the token has the correct scope for the `hpfracc` project
- Check that `.pypirc` is in your home directory (`~/.pypirc`)

## Current Status
- ✅ Package successfully uploaded to PyPI as `hpfracc-1.3.1`
- ✅ Authentication working with project-scoped API token
- ✅ Automated uploads via `twine upload dist/*`
