from typing import Dict
import numpy as np
from .base import BaseOODDetector


class EnergyDetector(BaseOODDetector):
    """
    Phương pháp 2: Energy-based Out-of-Distribution Detection
    Tài liệu tham khảo: Liu et al. (NeurIPS 2020) - "Energy-based Out-of-distribution Detection".

    Công thức năng lượng tự do Helmholtz:
        E(x; f) = -T * log sum_{c=1}^C exp(f_c(x) / T)
    Điểm ID (quy ước cao = ID, thấp = OOD):
        S_Energy(x) = -E(x; f) = T * log sum_{c=1}^C exp(f_c(x) / T)
    """

    name = "Energy"

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Nhiệt độ T phải lớn hơn 0.")
        self.temperature = float(temperature)

    def score(self, test_cache: Dict[str, np.ndarray]) -> np.ndarray:
        logits = np.asarray(test_cache["logits"], dtype=np.float64)
        scaled = logits / self.temperature
        # Tính log-sum-exp ổn định số học
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        lse = np.squeeze(max_scaled, axis=1) + np.log(
            np.sum(np.exp(scaled - max_scaled), axis=1)
        )
        return self.temperature * lse
