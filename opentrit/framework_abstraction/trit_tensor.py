import numpy as np

class TritTensor:
    """
    框架无关的混合三值Tensor抽象
    支持对称(-1,0,1)/非对称三值模式，统一跨框架操作接口
    """
    def __init__(self, data, mode="symmetric", sparse_rate=0.8, backend="pytorch"):
        """
        Args:
            data: 原始数据（np.ndarray/torch.Tensor/tf.Tensor）
            mode: 三值模式 - "symmetric"（对称）/"asymmetric"（非对称）/"auto"（自动）
            sparse_rate: 非对称模式下的稀疏率（0占比）
            backend: 底层框架 - "pytorch"/"tensorflow"
        """
        self.mode = mode
        self.sparse_rate = sparse_rate
        self.backend = backend
        self.raw_data = self._convert_to_numpy(data)
        self.ternary_data = self._quantize_to_ternary()

    def _convert_to_numpy(self, data):
        """统一转换为numpy数组，消除框架差异"""
        if "torch" in str(type(data)):
            return data.detach().cpu().numpy()
        elif "tensorflow" in str(type(data)):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _quantize_to_ternary(self):
        """核心：将数据量化为三值（对称/非对称）"""
        if self.mode == "symmetric":
            # 对称三值量化：-1, 0, 1
            mean = np.mean(self.raw_data)
            std = np.std(self.raw_data)
            threshold = mean + std
            ternary = np.where(self.raw_data > threshold, 1, 
                               np.where(self.raw_data < -threshold, -1, 0))
        elif self.mode == "asymmetric":
            # 非对称三值量化：高稀疏性，0占比由sparse_rate控制
            sorted_data = np.sort(np.abs(self.raw_data).flatten())
            threshold = sorted_data[int(len(sorted_data) * self.sparse_rate)]
            ternary = np.where(self.raw_data > threshold, 1, 
                               np.where(self.raw_data < -threshold, -1, 0))
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        return ternary

    def to_backend(self):
        """转换为目标框架的Tensor"""
        if self.backend == "pytorch":
            import torch
            return torch.tensor(self.ternary_data)
        elif self.backend == "tensorflow":
            import tensorflow as tf
            return tf.convert_to_tensor(self.ternary_data)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def __repr__(self):
        return f"TritTensor(mode={self.mode}, shape={self.ternary_data.shape}, sparse_rate={self.sparse_rate})"
