# PyPI Release Guide

This guide explains how to release new versions of HPFRACC to PyPI using automated GitHub Actions.

---

## 🚀 Quick Release Process

### 1. Update Version Number

Edit `pyproject.toml`:

```toml
[project]
version = "2.1.0"  # Update this
```

### 2. Update CHANGELOG.md

Add release notes for the new version in `CHANGELOG.md`.

### 3. Commit Changes

```bash
git add .
git commit -m "Release v2.1.0: Intelligent backend selection"
git push origin main
```

### 4. Create GitHub Release

Go to GitHub → Releases → Create a new release:

- **Tag version:** `v2.1.0` (must start with 'v')
- **Release title:** `v2.1.0 - Intelligent Backend Selection`
- **Description:** Copy from CHANGELOG.md
- Click **Publish release**

**That's it!** GitHub Actions will automatically:
1. Run tests on Python 3.8, 3.9, 3.10, 3.11
2. Build the distribution packages
3. Publish to PyPI using trusted publishing

---

## 🔧 One-Time Setup

### Step 1: Configure PyPI Trusted Publishing

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - **PyPI Project Name:** `hpfracc`
   - **Owner:** `dave2k77` (your GitHub username)
   - **Repository name:** `fractional_calculus_library`
   - **Workflow name:** `pypi-publish.yml`
   - **Environment name:** `pypi`

3. Click **Add**

### Step 2: Configure GitHub Environment

1. Go to your GitHub repo → Settings → Environments
2. Create environment named `pypi`
3. Add protection rules (optional):
   - Required reviewers
   - Wait timer
   - Deployment branches (only main)

### Step 3: Verify Workflow Files

Ensure these files exist:
- `.github/workflows/pypi-publish.yml` ✓
- `.github/workflows/tests.yml` ✓

---

## 📋 Release Checklist

Before creating a release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with new version
- [ ] Run tests locally: `pytest tests/test_integration_*.py`
- [ ] Check code quality: `black hpfracc/ && isort hpfracc/`
- [ ] Build locally: `python -m build`
- [ ] Test install: `pip install dist/hpfracc-*.whl`
- [ ] Commit and push all changes
- [ ] Create GitHub release with tag `vX.Y.Z`
- [ ] Verify GitHub Actions workflow completes
- [ ] Check package on PyPI: https://pypi.org/project/hpfracc/

---

## 🧪 Testing Before Release

### Local Build Test

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Test install in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install dist/hpfracc-2.1.0-py3-none-any.whl
python -c "import hpfracc; print(hpfracc.__version__)"
deactivate
rm -rf test_env
```

### Test on TestPyPI (Optional)

Use manual workflow dispatch to publish to TestPyPI:

1. Go to Actions → Publish to PyPI → Run workflow
2. Select branch → Run
3. This publishes to test.pypi.org instead

Install from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hpfracc
```

---

## 🔄 Automated Workflows

### PyPI Publish Workflow (`pypi-publish.yml`)

**Triggers:**
- When a GitHub release is published
- Manual workflow dispatch (for TestPyPI)

**Jobs:**
1. **Test:** Run integration tests on Python 3.8-3.11
2. **Build:** Build distribution packages (wheel + sdist)
3. **Publish:** Upload to PyPI using trusted publishing

### CI Tests Workflow (`tests.yml`)

**Triggers:**
- Push to main or develop branches
- Pull requests to main or develop

**Jobs:**
1. **Test:** Run tests on multiple Python versions
2. **Lint:** Code quality checks (black, isort, flake8)

---

## 🐛 Troubleshooting

### Build Fails

**Issue:** Package build fails

**Solution:**
```bash
# Check pyproject.toml syntax
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"

# Ensure all dependencies are listed
pip install -e .[dev]
```

### Tests Fail in CI

**Issue:** Tests pass locally but fail in GitHub Actions

**Solution:**
- Check Python version compatibility
- Ensure all test dependencies in requirements
- Check for missing files in package

### PyPI Upload Fails

**Issue:** "Invalid or non-existent authentication information"

**Solution:**
- Verify trusted publishing is configured on PyPI
- Check GitHub environment name matches (`pypi`)
- Ensure workflow has `id-token: write` permission

### Version Already Exists

**Issue:** "File already exists" on PyPI

**Solution:**
- You cannot overwrite PyPI releases
- Increment version number in `pyproject.toml`
- Create new release with new version

---

## 📊 Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New features (backward compatible)
- **PATCH** version (0.0.X): Bug fixes (backward compatible)

### Examples

- `2.1.0` → `2.1.1`: Bug fixes only
- `2.1.0` → `2.2.0`: New features added
- `2.1.0` → `3.0.0`: Breaking changes

---

## 🎯 Best Practices

### Release Frequency

- **Patch releases:** As needed for critical bugs
- **Minor releases:** Every 2-4 weeks with new features
- **Major releases:** Every 6-12 months for breaking changes

### Pre-releases

For beta/alpha versions, use version suffixes:

```toml
version = "2.2.0a1"  # Alpha 1
version = "2.2.0b1"  # Beta 1
version = "2.2.0rc1" # Release candidate 1
```

### Git Tags

Always create annotated tags:

```bash
git tag -a v2.1.0 -m "Release v2.1.0: Intelligent backend selection"
git push origin v2.1.0
```

Or use GitHub's "Create release" interface (recommended).

---

## 📚 Additional Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publish](https://github.com/marketplace/actions/pypi-publish)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

---

## 🆘 Support

If you encounter issues:

1. Check GitHub Actions logs
2. Review this guide's troubleshooting section
3. Consult PyPI and GitHub documentation
4. Create an issue in the repository

---

**Last Updated:** October 27, 2025  
**Current Version:** 2.1.0  
**Status:** Production Ready

