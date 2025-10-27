#!/bin/bash

# PyPI Upload Script for HPFRACC v2.2.0
# Usage: ./upload_to_pypi.sh

echo "🚀 HPFRACC v2.2.0 PyPI Upload Script"
echo "====================================="

# Check if packages exist
if [ ! -f "dist/hpfracc-2.2.0-py3-none-any.whl" ] || [ ! -f "dist/hpfracc-2.2.0.tar.gz" ]; then
    echo "❌ Distribution packages not found. Run 'python -m build' first."
    exit 1
fi

echo "✅ Found distribution packages:"
ls -la dist/

# Check for API token
if [ -z "$TWINE_PASSWORD" ]; then
    echo ""
    echo "⚠️  PyPI API token not set!"
    echo ""
    echo "To upload to PyPI:"
    echo "1. Get API token from: https://pypi.org/manage/account/token/"
    echo "2. Set environment variables:"
    echo "   export TWINE_USERNAME='__token__'"
    echo "   export TWINE_PASSWORD='your-api-token-here'"
    echo ""
    echo "3. Run this script again:"
    echo "   ./upload_to_pypi.sh"
    echo ""
    echo "Or use GitHub Actions by creating a release on GitHub."
    exit 1
fi

if [ -z "$TWINE_USERNAME" ]; then
    echo "Setting TWINE_USERNAME to '__token__'"
    export TWINE_USERNAME='__token__'
fi

echo ""
echo "📦 Uploading to PyPI..."
echo "Username: $TWINE_USERNAME"
echo "Password: [HIDDEN]"
echo ""

# Upload with non-interactive flag
twine upload --non-interactive dist/*

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Successfully uploaded HPFRACC v2.2.0 to PyPI!"
    echo "📦 Package URL: https://pypi.org/project/hpfracc/"
    echo ""
    echo "Test installation:"
    echo "pip install --upgrade hpfracc"
    echo "python -c \"import hpfracc; print(f'HPFRACC version: {hpfracc.__version__}')\""
else
    echo ""
    echo "❌ Upload failed. Check your API token and try again."
    exit 1
fi
