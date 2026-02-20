import torch
import torch.optim as optim

class TernaryAdamW(optim.AdamW):
    """
    适配混合三值模型的AdamW优化器：集成S-STE/Sp-STE梯度计算
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, amsgrad=False, sparse_lambda=0.03):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self.sparse_lambda = sparse_lambda  # 稀疏惩罚项

    def step(self, closure=None):
        """重载step：自定义梯度更新逻辑"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # 区分对称/非对称层，应用不同STE梯度
                if "symmetric" in p.name if hasattr(p, "name") else "conv" in p.name:
                    # S-STE：对称三值梯度
                    grad = torch.where((p.data == 1) | (p.data == -1), torch.ones_like(grad), 
                                       torch.where(p.data == 0, 0.5 * torch.ones_like(grad), grad))
                else:
                    # Sp-STE：非对称稀疏梯度 + 稀疏惩罚
                    grad = torch.where(p.data != 0, grad, torch.zeros_like(grad))
                    grad += self.sparse_lambda * p.data  # 稀疏惩罚

                p.grad.data = grad

        return super().step(closure)

class TernarySGD(optim.SGD):
    """适配混合三值模型的SGD优化器（简化版）"""
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, sparse_lambda=0.03):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.sparse_lambda = sparse_lambda

    def step(self, closure=None):
        # 逻辑同TernaryAdamW，略
        return super().step(closure)
