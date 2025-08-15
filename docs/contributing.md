# Contributing to Fractional Calculus Library

Thank you for your interest in contributing to the Fractional Calculus Library! This document provides guidelines and information for contributors.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style and Standards](#code-style-and-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation Standards](#documentation-standards)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)
8. [Code Review Process](#code-review-process)
9. [Release Process](#release-process)
10. [Community Guidelines](#community-guidelines)

---

## Getting Started

### Before You Start

1. **Check Existing Issues**: Search existing issues to avoid duplicates
2. **Read Documentation**: Familiarize yourself with the codebase
3. **Join Discussions**: Participate in GitHub discussions
4. **Start Small**: Begin with documentation or simple bug fixes

### Types of Contributions

We welcome various types of contributions:

- **Bug Fixes**: Fix existing issues
- **Feature Development**: Add new functionality
- **Documentation**: Improve guides, examples, and API docs
- **Testing**: Add tests or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Create new usage examples
- **Research**: Implement new algorithms or methods

---

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# 1. Fork the repository
# Go to https://github.com/dave2k77/fractional_calculus_library
# Click "Fork" button

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/fractional_calculus_library.git
cd fractional_calculus_library

# 3. Add upstream remote
git remote add upstream https://github.com/dave2k77/fractional_calculus_library.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 6. Install in development mode
pip install -e .

# 7. Install pre-commit hooks
pre-commit install
```

### Verifying Setup

```bash
# Run tests to verify installation
python scripts/run_tests.py --type fast

# Run code quality checks
python scripts/run_tests.py --no-coverage --reports
```

---

## Code Style and Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# ✅ Good
def compute_fractional_derivative(function_values, time_points, alpha):
    """Compute fractional derivative using specified method."""
    if alpha <= 0 or alpha >= 2:
        raise ValueError("Alpha must be in (0, 2)")
    
    return result

# ❌ Bad
def computeFractionalDerivative(f,t,a):
    if a<=0 or a>=2:
        raise ValueError("Alpha must be in (0, 2)")
    return result
```

### Naming Conventions

- **Functions and Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private Methods**: `_leading_underscore`
- **Protected Methods**: `_leading_underscore`

### Type Hints

Always use type hints for function signatures:

```python
from typing import Union, List, Optional
import numpy as np

def compute_derivative(
    function_values: np.ndarray,
    time_points: np.ndarray,
    alpha: float,
    method: str = "caputo"
) -> np.ndarray:
    """Compute fractional derivative."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def compute_caputo_derivative(
    function_values: np.ndarray,
    time_points: np.ndarray,
    alpha: float
) -> np.ndarray:
    """Compute Caputo fractional derivative.
    
    Args:
        function_values: Array of function values f(t)
        time_points: Array of time points t
        alpha: Fractional order (0 < alpha < 2)
        
    Returns:
        Array of derivative values D^α f(t)
        
    Raises:
        ValueError: If alpha is not in valid range
        ValueError: If arrays have different lengths
        
    Example:
        >>> t = np.linspace(0.1, 2.0, 100)
        >>> f = t**2
        >>> result = compute_caputo_derivative(f, t, 0.5)
    """
    pass
```

---

## Testing Guidelines

### Test Structure

Tests should be organized in the `tests/` directory:

```
tests/
├── test_algorithms/          # Algorithm tests
├── test_core/               # Core functionality tests
├── test_optimisation/       # Optimization tests
├── test_solvers/           # Solver tests
├── test_utils/             # Utility tests
├── test_validation/        # Validation tests
└── integration_tests/      # Integration tests
```

### Writing Tests

```python
import pytest
import numpy as np
from src.algorithms.caputo import CaputoDerivative

class TestCaputoDerivative:
    """Test suite for Caputo derivative implementation."""
    
    def test_basic_computation(self):
        """Test basic Caputo derivative computation."""
        # Arrange
        alpha = 0.5
        caputo = CaputoDerivative(alpha)
        t = np.linspace(0.1, 2.0, 100)
        f = t**2
        h = t[1] - t[0]
        
        # Act
        result = caputo.compute(f, t, h)
        
        # Assert
        assert result.shape == t.shape
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_invalid_alpha(self):
        """Test that invalid alpha values raise ValueError."""
        with pytest.raises(ValueError, match="Alpha must be in"):
            CaputoDerivative(-0.5)
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset (marked as slow)."""
        # Test implementation
        pass
```

### Test Categories

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test computational efficiency
- **Accuracy Tests**: Test numerical accuracy
- **Edge Case Tests**: Test boundary conditions

### Running Tests

```bash
# Run all tests
python scripts/run_tests.py

# Run specific test categories
python scripts/run_tests.py --type unit
python scripts/run_tests.py --type integration
python scripts/run_tests.py --type fast

# Run with pytest directly
pytest tests/ -v
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ --cov=src      # With coverage
```

### Test Coverage

Maintain high test coverage:
- **Minimum**: 80% overall coverage
- **Target**: 90%+ overall coverage
- **Critical modules**: 95%+ coverage

---

## Documentation Standards

### Code Documentation

- **All public functions**: Must have docstrings
- **All classes**: Must have class docstrings
- **Complex algorithms**: Include mathematical explanations
- **Examples**: Provide usage examples in docstrings

### API Documentation

Document all public APIs:

```python
class CaputoDerivative:
    """Caputo fractional derivative implementation.
    
    The Caputo derivative is defined as:
    
    D^α f(t) = (1/Γ(1-α)) ∫₀ᵗ (t-τ)^(-α) f'(τ) dτ
    
    This implementation uses numerical quadrature methods.
    
    Attributes:
        alpha: Fractional order (0 < alpha < 2)
        method: Quadrature method ("trapezoidal", "simpson", "gauss")
        grid_size: Number of grid points for discretization
    """
```

### User Documentation

- **Installation Guide**: Clear setup instructions
- **User Guide**: Comprehensive usage examples
- **API Reference**: Auto-generated from docstrings
- **Examples**: Practical code examples
- **Tutorials**: Step-by-step guides

---

## Pull Request Process

### Before Submitting

1. **Update your fork**: Sync with upstream
2. **Create feature branch**: `git checkout -b feature/your-feature`
3. **Make changes**: Follow coding standards
4. **Add tests**: Include tests for new functionality
5. **Update documentation**: Update relevant docs
6. **Run tests**: Ensure all tests pass
7. **Check quality**: Run code quality checks

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(algorithms): add Caputo-Fabrizio derivative implementation

fix(core): resolve memory leak in parallel computing

docs(user_guide): add GPU acceleration examples

test(validation): add convergence tests for new methods
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Code coverage maintained
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Related Issues
Closes #123
```

---

## Issue Reporting

### Bug Reports

Use the bug report template:

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Library Version: [e.g., 0.1.0]

## Additional Information
Screenshots, error messages, etc.
```

### Feature Requests

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed

## Proposed Implementation
Optional: How you think it should be implemented

## Alternatives Considered
Other approaches you've considered
```

---

## Code Review Process

### Review Checklist

Reviewers should check:

- [ ] **Functionality**: Code works as intended
- [ ] **Tests**: Adequate test coverage
- [ ] **Documentation**: Clear and complete
- [ ] **Performance**: No performance regressions
- [ ] **Security**: No security vulnerabilities
- [ ] **Style**: Follows coding standards
- [ ] **Maintainability**: Code is readable and maintainable

### Review Comments

Be constructive and specific:

```markdown
✅ Good: "Consider using numpy's vectorized operations here for better performance"

❌ Bad: "This is wrong"
```

### Review Timeline

- **Initial review**: Within 3-5 business days
- **Follow-up reviews**: Within 1-2 business days
- **Final approval**: Requires at least one maintainer approval

---

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes (1.0.0 → 2.0.0)
- **Minor**: New features (1.0.0 → 1.1.0)
- **Patch**: Bug fixes (1.0.0 → 1.0.1)

### Release Checklist

Before each release:

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version number updated
- [ ] Release notes prepared
- [ ] Tag created
- [ ] PyPI package updated (when applicable)

### Release Steps

```bash
# 1. Update version
# Edit setup.py or pyproject.toml

# 2. Update changelog
# Add release notes to CHANGELOG.md

# 3. Create release branch
git checkout -b release/v1.0.0

# 4. Final testing
python scripts/run_tests.py

# 5. Create tag
git tag -a v1.0.0 -m "Release v1.0.0"

# 6. Push changes
git push origin release/v1.0.0
git push origin v1.0.0

# 7. Create GitHub release
# Go to GitHub and create release from tag
```

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat others with respect
- **Be constructive**: Provide helpful feedback
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be patient**: Understand that contributors have different experience levels

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions
- **Email**: For sensitive matters (contact maintainers directly)

### Recognition

Contributors will be recognized in:

- **README.md**: List of contributors
- **Release notes**: Credit for significant contributions
- **Documentation**: Attribution for major features
- **GitHub**: Contributor statistics and profile

---

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory
- **Examples**: Review `examples/` directory
- **Issues**: Search existing issues
- **Discussions**: Join GitHub discussions

### Contact

- **Maintainers**: @dave2k77
- **Email**: [Project email]
- **Discussions**: [GitHub Discussions]

---

## Acknowledgments

Thank you for contributing to the Fractional Calculus Library! Your contributions help make this project better for everyone in the scientific computing community.

---

**Note**: These guidelines are living documents. They will be updated as the project evolves. If you have suggestions for improvements, please open an issue or discussion.
