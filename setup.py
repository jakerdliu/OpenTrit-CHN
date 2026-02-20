from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="opentrit",
    version="1.0.0",
    author="Your Name/Team Name",
    author_email="your-email@example.com",
    description="A Unified Framework for Hybrid Ternary Neural Networks with Cross-Framework Compatibility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/opentrit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "torch>=2.1.0",
        "tensorflow>=2.15.0",
        "onnx>=1.14.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "sphinx>=5.0.0",
            "black>=23.0.0",
        ]
    },
    keywords="ternary neural network, model compression, cross-framework, heterogeneous computing",
    project_urls={
        "Bug Reports": "https://github.com/your-username/opentrit/issues",
        "Documentation": "https://opentrit.readthedocs.io/",
        "Source Code": "https://github.com/your-username/opentrit",
    },
)
