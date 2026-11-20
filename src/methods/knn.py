from typing import Dict, List, Optional
import numpy as np
from .base import BaseOODDetector

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hóa vector đặc trưng lên mặt cầu đơn vị: z_hat = z / max(||z||_2, eps)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class DeepKNNDetector(BaseOODDetector):
    """
    Phương pháp 3 (Nhánh trọng tâm): Deep k-Nearest Neighbors (Deep kNN) cho OOD Detection.
    Tài liệu tham khảo: Sun et al. (ICML 2022) - "Out-of-Distribution Detection with Deep Nearest Neighbors".

    Quy trình toán học cốt lõi:
      1. Trích xuất đặc trưng tầng áp chót z = phi(x) in R^d.
      2. Chuẩn hóa L2: z_hat = z / ||z||_2 (nếu normalize_l2 = True).
      3. Lưu Feature Bank Z_N từ tập Train ID (có thể lấy mẫu tỷ lệ alpha in (0, 1]).
      4. Khi kiểm tra mẫu x*, tìm khoảng cách Euclid tới láng giềng gần thứ k trong Z_N:
             S_kNN(x*; k) = -||z_hat* - z_hat_(k)||_2
         (Giá trị càng lớn / càng gần 0 -> càng gần cụm dữ liệu ID -> mẫu Quen).
    """

    name = "Deep_kNN"

    def __init__(
        self,
        k: int = 50,
        normalize_l2: bool = True,
        sampling_ratio: float = 1.0,
        seed: int = 0,
        batch_size: int = 1024,
    ):
        if k < 1:
            raise ValueError("Tham số k phải >= 1.")
        if not (0.0 < sampling_ratio <= 1.0):
            raise ValueError("Tỷ lệ lấy mẫu sampling_ratio phải nằm trong (0, 1].")

        self.k = int(k)
        self.normalize_l2 = bool(normalize_l2)
        self.sampling_ratio = float(sampling_ratio)
        self.seed = int(seed)
        self.batch_size = int(batch_size)

        self.feature_bank: Optional[np.ndarray] = None
        self._bank_sq_norms: Optional[np.ndarray] = None
        self._faiss_index = None

    def fit(self, id_train_cache: Dict[str, np.ndarray]) -> "DeepKNNDetector":
        """Xây dựng Feature Bank từ tập đặc trưng huấn luyện ID (D_in^train)."""
        feats = np.asarray(id_train_cache["features"], dtype=np.float32)
        if feats.ndim != 2:
            raise ValueError(f"Expected 2D feature array (N, D), got shape {feats.shape}.")

        if self.normalize_l2:
            feats = l2_normalize(feats)

        n_total = feats.shape[0]
        if self.sampling_ratio < 1.0:
            n_keep = max(self.k, int(round(n_total * self.sampling_ratio)))
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(n_total, size=n_keep, replace=False)
            feats = feats[indices]

        if self.k > feats.shape[0]:
            raise ValueError(
                f"k={self.k} lớn hơn kích thước Feature Bank ({feats.shape[0]}). Hãy giảm k hoặc tăng sampling_ratio."
            )

        self.feature_bank = np.ascontiguousarray(feats, dtype=np.float32)

        if faiss is not None:
            dim = self.feature_bank.shape[1]
            self._faiss_index = faiss.IndexFlatL2(dim)
            self._faiss_index.add(self.feature_bank)
        else:
            self._faiss_index = None
            self._bank_sq_norms = np.sum(self.feature_bank ** 2, axis=1)

        return self

    def _query_top_k_distances(self, query_feats: np.ndarray, max_k: int) -> np.ndarray:
        """
        Trả về ma trận khoảng cách Euclid (N_query, max_k) tới max_k láng giềng gần nhất,
        đã sắp xếp tăng dần theo từng hàng.
        """
        if self.feature_bank is None:
            raise RuntimeError("DeepKNNDetector chưa được khởi tạo. Hãy gọi .fit(id_train_cache) trước.")

        q = np.asarray(query_feats, dtype=np.float32)
        if self.normalize_l2:
            q = l2_normalize(q)
        q = np.ascontiguousarray(q, dtype=np.float32)

        if self._faiss_index is not None:
            sq_dists, _ = self._faiss_index.search(q, max_k)
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.sqrt(sq_dists)

        # Fallback NumPy chuẩn xác & tối ưu theo từng lô (batch) khi chưa cài faiss
        assert self._bank_sq_norms is not None
        n_queries = q.shape[0]
        top_k_dists = np.empty((n_queries, max_k), dtype=np.float32)

        for start in range(0, n_queries, self.batch_size):
            end = min(start + self.batch_size, n_queries)
            q_batch = q[start:end]
            q_sq = np.sum(q_batch ** 2, axis=1, keepdims=True)
            # ||q - b||_2^2 = ||q||^2 + ||b||^2 - 2 * q @ b^T
            sq_dist_matrix = q_sq + self._bank_sq_norms[None, :] - 2.0 * (q_batch @ self.feature_bank.T)
            sq_dist_matrix = np.maximum(sq_dist_matrix, 0.0)

            # Dùng argpartition để lấy top max_k khoảng cách nhỏ nhất trong O(N_bank)
            part_idx = np.argpartition(sq_dist_matrix, kth=max_k - 1, axis=1)[:, :max_k]
            row_idx = np.arange(end - start)[:, None]
            part_sq = sq_dist_matrix[row_idx, part_idx]
            sorted_sq = np.sort(part_sq, axis=1)
            top_k_dists[start:end] = np.sqrt(sorted_sq)

        return top_k_dists

    def score(self, test_cache: Dict[str, np.ndarray]) -> np.ndarray:
        """Tính điểm S_kNN(x*) = -||z_hat* - z_hat_(k)||_2 tại láng giềng thứ k."""
        top_k_dists = self._query_top_k_distances(test_cache["features"], max_k=self.k)
        kth_dist = top_k_dists[:, self.k - 1]
        return -kth_dist.astype(np.float64)

    def score_multi_k(self, test_cache: Dict[str, np.ndarray], k_list: List[int]) -> Dict[int, np.ndarray]:
        """
        Tính điểm S_kNN(x*) cho nhiều giá trị k chỉ trong 1 lần truy vấn khoảng cách (dùng cho Ablation Study).
        """
        max_k = max(k_list)
        top_dists = self._query_top_k_distances(test_cache["features"], max_k=max_k)
        return {k: -top_dists[:, k - 1].astype(np.float64) for k in k_list}
