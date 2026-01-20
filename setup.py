"""Setup script for Agent-CAP."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="agent-cap",
    version="0.1.0",
    author="Agent-CAP Team",
    description="Benchmarking of Cost, Accuracy, and Performance for Agentic AI Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Auto-CAP/AgentCAP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies - minimal
    ],
    extras_require={
        "viz": [
            "plotly>=5.0.0",
        ],
        "cuda": [
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "all": [
            "plotly>=5.0.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent-cap=agent_cap.cli:main",
        ],
    },
)
