from typing import Dict, List
import numpy as np


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict[str, float]:
    """
    Tính 2 độ đo bắt buộc của Chủ đề 11: AUROC (%) và FPR@95 (%).

    Quy ước điểm số:
      - Điểm số S(x) CÀNG CAO -> Càng giống mẫu Quen (In-Distribution, nhãn dương = 1).
      - Điểm số S(x) CÀNG THẤP -> Càng giống mẫu Lạ (Out-of-Distribution, nhãn âm = 0).

    Args:
        id_scores: Mảng 1 chiều chứa điểm S(x) của các mẫu ID Test (N_id,).
        ood_scores: Mảng 1 chiều chứa điểm S(x) của các mẫu OOD Test (N_ood,).

    Returns:
        Dict chứa:
          - 'auroc': Diện tích dưới đường cong ROC (thang %, càng CAO càng tốt).
          - 'fpr95': Tỷ lệ mẫu OOD bị nhận nhầm là ID khi TPR(ID) = 95% (thang %, càng THẤP càng tốt).
          - 'threshold_95': Ngưỡng quyết định lambda tại TPR = 95%.
    """
    id_scores = np.asarray(id_scores, dtype=np.float64).ravel()
    ood_scores = np.asarray(ood_scores, dtype=np.float64).ravel()

    if id_scores.size == 0 or ood_scores.size == 0:
        raise ValueError("id_scores và ood_scores không được rỗng.")

    # 1. Tính ngưỡng lambda tại TPR = 95% trên tập ID (tức phân vị thứ 5% từ dưới lên của id_scores)
    threshold_95 = float(np.percentile(id_scores, 5.0))
    fpr95 = float(np.mean(ood_scores >= threshold_95) * 100.0)

    # 2. Tính AUROC bằng phương pháp hình thang trên đường cong ROC thực nghiệm
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])

    # Sắp xếp giảm dần theo điểm số
    desc_idx = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # Chỉ lấy các vị trí có giá trị điểm số thay đổi để xử lý chính xác các điểm bằng nhau (ties)
    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    # Bổ sung gốc tọa độ (0, 0)
    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    tpr = tps / tps[-1]
    fpr = fps / fps[-1]

    # Tích phân hình thang
    auroc = float(np.trapezoid(tpr, fpr) * 100.0) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr) * 100.0)

    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "threshold_95": threshold_95,
    }


def summarize_multi_seed(seed_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Tổng hợp mean ± std qua nhiều random seeds (Mục 4.2 của đề bài)."""
    aurocs = np.array([m["auroc"] for m in seed_metrics], dtype=np.float64)
    fpr95s = np.array([m["fpr95"] for m in seed_metrics], dtype=np.float64)
    latencies = np.array([m.get("latency_ms", 0.0) for m in seed_metrics], dtype=np.float64)

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs, ddof=0)),
        "fpr95_mean": float(np.mean(fpr95s)),
        "fpr95_std": float(np.std(fpr95s, ddof=0)),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies, ddof=0)),
    }
