import matplotlib.pyplot as plt
import numpy as np

def plot_tensor_distribution(tensor, save_path=None):
    """绘制三值张量分布热力图"""
    plt.figure(figsize=(8, 6))
    if hasattr(tensor, "ternary_data"):
        data = tensor.ternary_data
    else:
        data = tensor.detach().cpu().numpy() if "torch" in str(type(tensor)) else tensor.numpy()
    
    plt.hist(data.flatten(), bins=3, label=["-1", "0", "1"], color=["blue", "gray", "red"])
    plt.title("Hybrid Ternary Tensor Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def quantization_error_analysis(quantized_model, reference_model, save_path=None):
    """分析量化误差：计算与原始模型的L2距离"""
    quantized_params = list(quantized_model.parameters())
    ref_params = list(reference_model.parameters())
    errors = []

    for q_p, r_p in zip(quantized_params, ref_params):
        q_data = q_p.detach().cpu().numpy()
        r_data = r_p.detach().cpu().numpy()
        l2_error = np.linalg.norm(q_data - r_data) / np.linalg.norm(r_data)
        errors.append(l2_error)

    # 绘制误差分布
    plt.figure(figsize=(10, 5))
    plt.plot(errors, marker="o", label="Layer-wise L2 Error")
    plt.axhline(np.mean(errors), color="red", linestyle="--", label=f"Mean Error: {np.mean(errors):.4f}")
    plt.title("Quantization Error Analysis")
    plt.xlabel("Layer Index")
    plt.ylabel("Normalized L2 Error")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"mean_error": np.mean(errors), "max_error": np.max(errors), "layer_errors": errors}
