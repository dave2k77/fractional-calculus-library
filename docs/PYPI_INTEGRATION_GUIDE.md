# 📦 PyPI Integration Guide

This guide explains how to link your ReadTheDocs documentation to your PyPI package for seamless user experience.

## 🔗 **How PyPI-Documentation Linking Works**

When users visit your package on PyPI, they can click on the "Documentation" link to go directly to your ReadTheDocs site. This creates a professional, integrated experience.

## 📋 **Current Setup Status**

### ✅ **Already Configured**

Your `pyproject.toml` already includes the necessary metadata:

```toml
[project.urls]
Homepage = "https://github.com/dave2k77/fractional_calculus_library"
Repository = "https://github.com/dave2k77/fractional_calculus_library.git"
Documentation = "https://fractional-calculus-library.readthedocs.io"
"Bug Tracker" = "https://github.com/dave2k77/fractional_calculus_library/issues"
"Source Code" = "https://github.com/dave2k77/fractional_calculus_library"
"Download" = "https://github.com/dave2k77/fractional_calculus_library/releases"
"Academic Contact" = "mailto:d.r.chin@pgr.reading.ac.uk"
```

### 🎯 **What This Achieves**

1. **PyPI Package Page**: Shows "Documentation" link in the sidebar
2. **Direct Navigation**: Users can click to go to ReadTheDocs
3. **Professional Appearance**: Integrated documentation experience
4. **SEO Benefits**: Better discoverability and linking

## 🚀 **Steps to Update PyPI**

### **1. Build New Distribution**

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build source and wheel distributions
python -m build

# Verify the build
ls -la dist/
```

### **2. Upload to PyPI**

```bash
# Upload to PyPI (replace with your credentials)
python -m twine upload dist/*

# Or upload to TestPyPI first for testing
python -m twine upload --repository testpypi dist/*
```

### **3. Verify the Update**

1. **Visit PyPI**: Go to https://pypi.org/project/hpfracc/
2. **Check Sidebar**: Look for "Documentation" link
3. **Test Link**: Click to ensure it goes to ReadTheDocs
4. **Verify Metadata**: Check all URLs are correct

## 🔧 **Configuration Details**

### **PyPI Package Metadata**

The following metadata appears on your PyPI page:

- **Project Name**: `hpfracc`
- **Version**: `1.1.2`
- **Description**: High-Performance Fractional Calculus Library with Machine Learning Integration and Graph Neural Networks
- **Author**: Davian R. Chin (d.r.chin@pgr.reading.ac.uk)
- **License**: MIT
- **Documentation**: https://fractional-calculus-library.readthedocs.io

### **ReadTheDocs Configuration**

Your `.readthedocs.yml` ensures:
- ✅ **Automatic builds** on GitHub pushes
- ✅ **Documentation hosting** at the correct URL
- ✅ **Version management** for different releases
- ✅ **Search functionality** across all documentation

## 📊 **User Experience Flow**

### **For End Users**

1. **Discover Package**: User finds `hpfracc` on PyPI
2. **View Details**: Sees comprehensive metadata and description
3. **Access Documentation**: Clicks "Documentation" link
4. **Read Docs**: Lands on ReadTheDocs with full documentation
5. **Install Package**: Uses `pip install hpfracc` or `pip install hpfracc[ml]`

### **For Developers**

1. **Clone Repository**: `git clone https://github.com/dave2k77/fractional_calculus_library.git`
2. **Install Development**: `pip install -e .[dev]`
3. **Access Docs**: Visit ReadTheDocs for development guides
4. **Contribute**: Follow development guidelines in documentation

## 🎨 **Customization Options**

### **Additional PyPI Metadata**

You can add more metadata to enhance your PyPI page:

```toml
[project.urls]
# Existing URLs...
"Changelog" = "https://github.com/dave2k77/fractional_calculus_library/blob/main/CHANGELOG.md"
"Funding" = "https://github.com/sponsors/dave2k77"
"Say Thanks!" = "https://saythanks.io/to/d.r.chin@pgr.reading.ac.uk"
```

### **Enhanced Classifiers**

Add more classifiers for better categorization:

```toml
classifiers = [
    # Existing classifiers...
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Medical Science Apps",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Visualization",
    "Topic :: Software Development :: Libraries :: Application Frameworks",
    "Topic :: Software Development :: Testing",
    "Topic :: Software Development :: Quality Assurance",
]
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Documentation Link Not Working**
   - Verify ReadTheDocs URL is correct
   - Check that ReadTheDocs build is successful
   - Ensure the URL is accessible

2. **PyPI Update Not Reflecting**
   - Wait a few minutes for PyPI to update
   - Clear browser cache
   - Check PyPI CDN propagation

3. **ReadTheDocs Build Failures**
   - Check build logs on ReadTheDocs
   - Verify `.readthedocs.yml` configuration
   - Ensure all dependencies are available

### **Verification Commands**

```bash
# Check PyPI package info
pip show hpfracc

# Verify documentation URL
curl -I https://fractional-calculus-library.readthedocs.io

# Test package installation
pip install hpfracc --force-reinstall
```

## 📈 **Analytics and Monitoring**

### **PyPI Statistics**

- **Downloads**: Track package usage on PyPI
- **Views**: Monitor PyPI page visits
- **Documentation Clicks**: Track ReadTheDocs referrals

### **ReadTheDocs Analytics**

- **Page Views**: Monitor documentation usage
- **Search Queries**: Understand user needs
- **Popular Pages**: Identify most-used documentation

## 🎯 **Best Practices**

### **Documentation Maintenance**

1. **Keep Documentation Updated**: Sync with code changes
2. **Version Documentation**: Tag releases appropriately
3. **Monitor Build Status**: Ensure ReadTheDocs builds succeed
4. **User Feedback**: Collect and address documentation issues

### **PyPI Maintenance**

1. **Regular Updates**: Keep package metadata current
2. **Version Management**: Follow semantic versioning
3. **Release Notes**: Provide clear changelog
4. **Quality Assurance**: Test before uploading

## 🔗 **Useful Links**

- **PyPI Package**: https://pypi.org/project/hpfracc/
- **ReadTheDocs**: https://fractional-calculus-library.readthedocs.io
- **GitHub Repository**: https://github.com/dave2k77/fractional_calculus_library
- **PyPI Upload Guide**: https://packaging.python.org/tutorials/packaging-projects/
- **ReadTheDocs Guide**: https://docs.readthedocs.io/

## 📞 **Support**

For issues with PyPI integration:
- **PyPI Issues**: Contact PyPI administrators
- **ReadTheDocs Issues**: Check ReadTheDocs documentation
- **Package Issues**: Use GitHub issues for package-specific problems

---

**✅ Your PyPI-Documentation integration is properly configured and ready to provide users with a seamless experience!**
