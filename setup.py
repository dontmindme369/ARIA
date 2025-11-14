#!/usr/bin/env python3
"""
ARIA - Adaptive Retrieval with Intelligent Anchoring
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="aria-rag",
    version="1.0.0",
    description="Adaptive RAG system with quaternion exploration and multi-armed bandits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/dontmindme369/aria",
    license="MIT",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include data files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
    ],
    
    # Optional dependencies
    extras_require={
        "semantic": ["sentence-transformers>=2.2.0"],
        "pdf": ["PyPDF2>=3.0.0"],
        "docx": ["python-docx>=0.8.11"],
        "all": [
            "sentence-transformers>=2.2.0",
            "PyPDF2>=3.0.0",
            "python-docx>=0.8.11",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.990",
        ],
    },
    
    # CLI entry points
    entry_points={
        "console_scripts": [
            "aria=aria_main:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    
    # Keywords for discoverability
    keywords="rag retrieval nlp machine-learning adaptive-systems multi-armed-bandits",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/dontmindme369/aria/issues",
        "Source": "https://github.com/dontmindme369/aria",
        "Documentation": "https://github.com/dontmindme369/aria/blob/main/README.md",
    },
)
