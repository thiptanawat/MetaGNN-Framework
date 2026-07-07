from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="metagnn",
    version="1.0.0",
    author="Thiptanawat Phongwattana",
    author_email="thiptanawat@example.com",
    description="Heterogeneous GATv2 framework for metabolic reaction activity scoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/metagnn",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "torch-geometric>=2.5.0",
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.17",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.1",
        "cobra>=0.29.0",
        "rdkit>=2023.09.5",
        "networkx",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4",
        "tqdm>=4.62.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "agent": [
            "vllm>=0.6.0",
            "langgraph>=0.1.0",
            "langchain>=0.1.0",
            "faiss-cpu>=1.8.0",
            "sentence-transformers>=2.2.0",
            "mygene>=3.2.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
)
