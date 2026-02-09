"""
OpenTrit: A Unified Framework for Hybrid Ternary Neural Networks with Cross-Framework Compatibility
GitHub: https://github.com/your-username/opentrit
Documentation: https://opentrit.readthedocs.io/
"""

__version__ = "$PROJECT_VERSION"
__author__ = "$AUTHOR_NAME"
__license__ = "MIT"

# 暴露核心API
from opentrit.framework_abstraction.trit_tensor import TritTensor
from opentrit.dynamic_ternary_engine.adaptive_switch import quantize
from opentrit.heterogeneous_scheduler import HeterogeneousScheduler
from opentrit.toolchain import export, import_onnx
from opentrit.optim import TernaryAdamW, TernarySGD
from opentrit.viewer import plot_tensor_distribution, quantization_error_analysis
