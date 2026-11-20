from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BaseOODDetector(ABC):
    """
    Giao diện chuẩn cho mọi phương pháp Post-hoc OOD Detection trong Chủ đề 11.
    Quy ước chung:
      - score(cache) trả về mảng 1 chiều S(x) với quy ước:
        S(x) CÀNG CAO -> Càng chắc chắn là mẫu Quen (In-Distribution).
        S(x) CÀNG THẤP -> Càng nghi ngờ là mẫu Lạ (Out-of-Distribution).
    """

    name: str = "BaseDetector"

    def fit(self, id_train_cache: Dict[str, np.ndarray]) -> "BaseOODDetector":
        """
        Khởi tạo tham số từ tập Train ID (nếu cần).
        - MSP và Energy không cần dùng id_train_cache (stateless).
        - Deep kNN dùng id_train_cache['features'] để xây dựng Feature Bank.
        """
        return self

    @abstractmethod
    def score(self, test_cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Tính điểm tin cậy ID S(x) cho từng mẫu trong test_cache."""
        raise NotImplementedError
