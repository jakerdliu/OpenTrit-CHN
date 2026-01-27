# OpenTrit-
• OpenTrit, an open-source cross-framework mixed ternary toolkit, supports one-click conversion of mixed ternary models between PyTorch and TensorFlow. It encapsulates heterogeneous computing power scheduling and quantization optimization, addressing the issues of "framework dependency and poor usability" present in existing ternary tools.
# OpenTrit
[![PyPI Version](https://img.shields.io/pypi/v/opentrit)](https://pypi.org/project/opentrit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/opentrit)](https://pypi.org/project/opentrit/)
[![Documentation Status](https://readthedocs.org/projects/opentrit/badge/?version=latest)](https://opentrit.readthedocs.io/en/latest/)

A Unified Framework for Hybrid Ternary Neural Networks with Cross-Framework Compatibility

## 🔥 Core Features
- **Cross-Framework Compatibility**: Native support for PyTorch/TensorFlow/TensorRT, "one-code-multi-framework" deployment.
- **Dynamic Hybrid Ternary Engine**: Adaptive symmetric-asymmetric switching + hierarchical error compensation.
- **Heterogeneous Computing**: Optimized task allocation across CPU/GPU/NPU/FPGA.
- **Full-Stack Toolchain**: Model conversion, training, debugging, deployment in one package.
- **High Efficiency**: 16x model compression, 3-4x inference speedup, <1% accuracy loss.

## 🚀 Quick Start
### Installation
```bash
# Install from PyPI
pip install opentrit

# Or install from source
git clone https://github.com/your-username/opentrit.git
cd opentrit
pip install -e .
