#!/usr/bin/env python3
"""
Setup script for Fractional Calculus Library

A comprehensive library for numerical methods in fractional calculus
with JAX and NUMBA optimizations.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fractional-calculus-library",
    version="0.1.0",
    author="David",
    author_email="dave2k77@gmail.com",
    description="Optimized numerical methods for fractional calculus using JAX and NUMBA",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dave2k77/fractional_calculus_library",
    project_urls={
        "Bug Tracker": "https://github.com/dave2k77/fractional_calculus_library/issues",
        "Documentation": "https://fractional-calculus-library.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-benchmark>=3.4.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
