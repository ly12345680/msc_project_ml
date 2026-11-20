import os
import platform
import random
from typing import Dict, Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def set_seed(seed: int = 0) -> None:
    """Thiết lập cố định random seed cho Python, NumPy và PyTorch để tái lập kết quả."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_hardware_info() -> Dict[str, Any]:
    """Ghi nhận thông tin phần cứng (CPU/GPU) theo yêu cầu Mục 4.2 của đề bài."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "device": "cpu",
        "gpu_name": "N/A",
    }
    if torch is not None:
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["gpu_name"] = "Apple Silicon MPS"
    return info
