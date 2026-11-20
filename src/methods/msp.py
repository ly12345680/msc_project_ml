from typing import Dict
import numpy as np
from .base import BaseOODDetector


class MSPDetector(BaseOODDetector):
    """
    Phương pháp 1 (Baseline): Maximum Softmax Probability (MSP)
    Tài liệu tham khảo: Hendrycks & Gimpel (ICLR 2017) - "A Baseline for Detecting Misclassified
    and Out-of-Distribution Examples in Neural Networks".

    Công thức:
        S_MSP(x) = max_c [ exp(f_c(x)) / sum_j exp(f_j(x)) ]
    """

    name = "MSP"

    def score(self, test_cache: Dict[str, np.ndarray]) -> np.ndarray:
        logits = np.asarray(test_cache["logits"], dtype=np.float64)
        # Trừ max logit để tránh tràn số (numerical stability) khi tính exp
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.max(probs, axis=1)
