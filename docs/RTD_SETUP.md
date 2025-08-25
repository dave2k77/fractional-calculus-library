# ReadTheDocs Setup Guide for HPFRACC

This guide will walk you through setting up ReadTheDocs for the HPFRACC project.

## 🚀 Quick Start

### 1. Prerequisites

- GitHub repository with HPFRACC code
- GitHub account with admin access to the repository
- ReadTheDocs account (free at [readthedocs.org](https://readthedocs.org))

### 2. Repository Setup

The repository should have the following structure:
```
fractional_calculus_library/
├── .readthedocs.yml          # RTD configuration
├── docs/
│   ├── conf.py              # Sphinx configuration
│   ├── index.md             # Main documentation page
│   ├── requirements.txt     # Documentation dependencies
│   ├── Makefile            # Build commands
│   ├── _static/            # Static files (CSS, images)
│   └── _templates/         # Custom templates
├── hpfracc/                # Source code
├── pyproject.toml          # Project configuration
└── README.md               # Project description
```

## 🔧 ReadTheDocs Configuration

### 1. Create ReadTheDocs Account

1. Go to [readthedocs.org](https://readthedocs.org)
2. Sign up with your GitHub account
3. Authorize ReadTheDocs to access your repositories

### 2. Import Your Project

1. Click "Import a Project"
2. Select your GitHub repository (`fractional_calculus_library`)
3. Choose the default settings
4. Click "Create"

### 3. Configure Build Settings

In your ReadTheDocs project settings:

#### Build Configuration
- **Documentation Type**: Sphinx
- **Python Interpreter**: Python 3.9
- **Install Project**: Yes
- **Use system packages**: No

#### Advanced Settings
- **Default Branch**: `main`
- **Default Version**: `latest`
- **Show version warning**: Yes
- **Privacy**: Public (recommended for open source)

## 📚 Documentation Structure

### 1. Main Pages

- **`index.md`**: Landing page with overview and quick start
- **`user_guide.md`**: Getting started and basic usage
- **`api_reference.md`**: Complete API documentation
- **`examples.md`**: Code examples and tutorials
- **`model_theory.md`**: Mathematical foundations

### 2. API Documentation

- **`api_reference/advanced_methods_api.md`**: Advanced algorithms
- **Auto-generated**: Core modules and classes

### 3. Static Assets

- **`_static/custom.css`**: Custom styling
- **`_static/logo.png`**: Project logo (optional)
- **`_static/favicon.ico`**: Browser favicon (optional)

## 🛠️ Local Development

### 1. Install Dependencies

```bash
cd docs
pip install -r requirements.txt
```

### 2. Build Documentation

```bash
# Quick build
make build

# Build and serve locally
make serve

# Build all formats
make all

# Clean build directory
make clean
```

### 3. Preview Changes

After building, open `_build/html/index.html` in your browser to preview the documentation.

## 🔄 Continuous Integration

### 1. GitHub Actions (Optional)

Create `.github/workflows/docs.yml`:

```yaml
name: Build Documentation

on:
  push:
    branches: [ main ]
    paths: [ 'docs/**', 'hpfracc/**', 'pyproject.toml' ]
  pull_request:
    branches: [ main ]
    paths: [ 'docs/**', 'hpfracc/**', 'pyproject.toml' ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r docs/requirements.txt
        pip install -e .
    - name: Build documentation
      run: |
        cd docs
        make build
    - name: Check for broken links
      run: |
        cd docs
        make linkcheck
```

### 2. Automatic Deployment

ReadTheDocs will automatically:
- Build documentation on every push to `main`
- Create version-specific documentation
- Handle pull request previews

## 🎨 Customization

### 1. Theme Options

Modify `docs/conf.py`:

```python
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
}
```

### 2. Custom CSS

Add custom styles in `docs/_static/custom.css`:

```css
:root {
    --hpfracc-primary: #2980B9;
    --hpfracc-secondary: #3498DB;
    --hpfracc-accent: #E74C3C;
}
```

### 3. Custom Templates

Create custom templates in `docs/_templates/` for advanced customization.

## 📖 Writing Documentation

### 1. Markdown Support

The documentation uses MyST-Parser for enhanced Markdown:

```markdown
# Headers
## Subheaders

```python
# Code blocks
def example():
    return "Hello, World!"
```

::: tip
**Tip**: Use admonitions for important information
:::

::: warning
**Warning**: Important warnings go here
:::
```

### 2. Math Equations

Use LaTeX math with MyST:

```markdown
$$
\frac{d^\alpha f(x)}{dx^\alpha} = \lim_{h \to 0} \frac{1}{h^\alpha} \sum_{k=0}^{\infty} (-1)^k \binom{\alpha}{k} f(x - kh)
$$
```

### 3. API Documentation

Use Sphinx autodoc for automatic API documentation:

```python
def fractional_derivative(f, alpha, method='riemann_liouville'):
    """
    Compute the fractional derivative of function f.
    
    Args:
        f: Function to differentiate
        alpha: Fractional order
        method: Differentiation method
        
    Returns:
        Fractional derivative function
    """
    pass
```

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for missing dependencies
   - Verify Python version compatibility
   - Check for syntax errors in Markdown files

2. **Missing Modules**
   - Ensure `PYTHONPATH` is set correctly
   - Install the package in development mode: `pip install -e .`

3. **Styling Issues**
   - Verify CSS file paths
   - Check for CSS syntax errors
   - Clear browser cache

4. **Math Rendering**
   - Ensure MathJax is configured correctly
   - Check LaTeX syntax in equations

### Debug Commands

```bash
# Build with verbose output
sphinx-build -v docs _build/html

# Build with warnings as errors
sphinx-build -W docs _build/html

# Check for broken links
sphinx-build -b linkcheck docs _build/linkcheck
```

## 🔗 Useful Links

- [ReadTheDocs Documentation](https://docs.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [MyST-Parser Documentation](https://myst-parser.readthedocs.io/)
- [Sphinx RTD Theme](https://sphinx-rtd-theme.readthedocs.io/)

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review ReadTheDocs build logs
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Documenting! 📚✨**
