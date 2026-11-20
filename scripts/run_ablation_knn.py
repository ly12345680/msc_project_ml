#!/usr/bin/env python3
"""
Script 3 (Dành riêng cho nhánh Deep kNN của bạn):
Thực hiện 3 nghiên cứu cắt giảm (Ablation Studies) theo configs/ablation_knn.yaml:
  1. Ảnh hưởng của số láng giềng k in {1, 5, 10, 20, 50, 100, 200, 500}.
  2. Vai trò quyết định của chuẩn hóa L2 (With L2-Norm vs. Without L2-Norm).
  3. Tỷ lệ lấy mẫu Feature Bank alpha in {1%, 5%, 10%, 25%, 50%, 100%} vs. Tốc độ truy vấn (Latency).

Cách chạy:
    python scripts/run_ablation_knn.py --config configs/ablation_knn.yaml
"""
import argparse
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import load_feature_cache
from src.methods.knn import DeepKNNDetector
from src.utils.metrics import compute_ood_metrics, summarize_multi_seed


def load_ablation_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return {
            "seeds": [0, 1, 2],
            "cache_dir": "./data/extracted_cache",
            "results_dir": "./results",
            "ood_datasets": {
                "near_ood": ["cifar100"],
                "far_ood": ["svhn", "mnist", "dtd"],
            },
            "ablations": {
                "k_sweep": {"k_values": [1, 5, 10, 20, 50, 100, 200, 500], "normalize_l2": True, "sampling_ratio": 1.0},
                "l2_norm_comparison": {"k": 50, "normalize_l2_options": [True, False], "sampling_ratio": 1.0},
                "sampling_ratio_sweep": {"k": 50, "normalize_l2": True, "ratios": [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]},
            },
        }


def run_ablation(config_path: str, cache_dir_override: str = None):
    cfg = load_ablation_config(config_path)
    seeds: List[int] = cfg.get("seeds", [0, 1, 2])
    cache_dir = cache_dir_override or cfg.get("cache_dir", "./data/extracted_cache")
    results_dir = cfg.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    near_ood = cfg["ood_datasets"].get("near_ood", ["cifar100"])
    far_ood = cfg["ood_datasets"].get("far_ood", ["svhn", "mnist", "dtd"])
    all_ood = near_ood + far_ood

    # Đọc toàn bộ cache vào bộ nhớ để chạy ablation cực nhanh
    caches = {}
    for seed in seeds:
        caches[seed] = {
            "id_train": load_feature_cache(cache_dir, seed, "cifar10_train"),
            "id_test": load_feature_cache(cache_dir, seed, "cifar10_test"),
            "ood": {name: load_feature_cache(cache_dir, seed, name) for name in all_ood},
        }

    # =========================================================================
    # ABLATION 1: Khảo sát tham số k (Số láng giềng gần nhất)
    # =========================================================================
    k_cfg = cfg["ablations"]["k_sweep"]
    k_values: List[int] = k_cfg["k_values"]
    print("\n================================================================================")
    print(" [ABLATION 1] ẢNH HƯỞNG CỦA SỐ LÁNG GIỀNG k TRONG DEEP kNN (Trung bình Near/Far-OOD)")
    print("================================================================================")
    print(f"{'k':<8} | {'Near-OOD AUROC (↑)':<20} | {'Near-OOD FPR95 (↓)':<20} | {'Far-OOD AUROC (↑)':<20} | {'Far-OOD FPR95 (↓)':<20}")
    print("-" * 98)

    k_records: Dict[int, Dict[str, List[Dict[str, float]]]] = {
        k: {"near": [], "far": []} for k in k_values
    }

    for seed in seeds:
        det = DeepKNNDetector(
            k=max(k_values),
            normalize_l2=k_cfg.get("normalize_l2", True),
            sampling_ratio=k_cfg.get("sampling_ratio", 1.0),
            seed=seed,
        ).fit(caches[seed]["id_train"])

        id_multi = det.score_multi_k(caches[seed]["id_test"], k_values)
        for ood_name in all_ood:
            group_key = "near" if ood_name in near_ood else "far"
            ood_multi = det.score_multi_k(caches[seed]["ood"][ood_name], k_values)
            for k in k_values:
                m = compute_ood_metrics(id_multi[k], ood_multi[k])
                k_records[k][group_key].append(m)

    for k in k_values:
        near_agg = summarize_multi_seed(k_records[k]["near"])
        far_agg = summarize_multi_seed(k_records[k]["far"])
        print(
            f"{k:<8} | "
            f"{near_agg['auroc_mean']:6.2f} ± {near_agg['auroc_std']:4.2f}%     | "
            f"{near_agg['fpr95_mean']:6.2f} ± {near_agg['fpr95_std']:4.2f}%     | "
            f"{far_agg['auroc_mean']:6.2f} ± {far_agg['auroc_std']:4.2f}%     | "
            f"{far_agg['fpr95_mean']:6.2f} ± {far_agg['fpr95_std']:4.2f}%"
        )

    # =========================================================================
    # ABLATION 2: Có Chuẩn hóa L2 vs. Không có Chuẩn hóa L2
    # =========================================================================
    l2_cfg = cfg["ablations"]["l2_norm_comparison"]
    k_fixed = l2_cfg.get("k", 50)
    print("\n================================================================================")
    print(f" [ABLATION 2] VAI TRÒ CỦA CHUẨN HÓA L2 (With vs. Without L2-Norm, k = {k_fixed})")
    print("================================================================================")
    print(f"{'L2-Norm':<10} | {'OOD Dataset':<12} | {'AUROC (↑ %)':<20} | {'FPR@95 (↓ %)':<20}")
    print("-" * 70)

    for norm_opt in l2_cfg.get("normalize_l2_options", [True, False]):
        for ood_name in all_ood:
            seed_list = []
            for seed in seeds:
                det = DeepKNNDetector(
                    k=k_fixed,
                    normalize_l2=norm_opt,
                    sampling_ratio=l2_cfg.get("sampling_ratio", 1.0),
                    seed=seed,
                ).fit(caches[seed]["id_train"])
                id_s = det.score(caches[seed]["id_test"])
                ood_s = det.score(caches[seed]["ood"][ood_name])
                seed_list.append(compute_ood_metrics(id_s, ood_s))
            agg = summarize_multi_seed(seed_list)
            label = "With L2" if norm_opt else "No L2"
            print(
                f"{label:<10} | {ood_name:<12} | "
                f"{agg['auroc_mean']:6.2f} ± {agg['auroc_std']:4.2f}%     | "
                f"{agg['fpr95_mean']:6.2f} ± {agg['fpr95_std']:4.2f}%"
            )

    # =========================================================================
    # ABLATION 3: Tỷ lệ lấy mẫu Feature Bank (alpha) vs. Tốc độ suy luận
    # =========================================================================
    sr_cfg = cfg["ablations"]["sampling_ratio_sweep"]
    ratios: List[float] = sr_cfg.get("ratios", [0.01, 0.05, 0.10, 0.25, 0.50, 1.00])
    print("\n================================================================================")
    print(" [ABLATION 3] TỶ LỆ LẤY MẪU FEATURE BANK (alpha) vs. ĐỘ CHÍNH XÁC & TỐC ĐỘ")
    print("================================================================================")
    print(f"{'Alpha (%)':<10} | {'Avg AUROC (↑ %)':<20} | {'Avg FPR@95 (↓ %)':<20} | {'Query Latency (ms)':<18}")
    print("-" * 76)

    for alpha in ratios:
        seed_metrics = []
        for seed in seeds:
            det = DeepKNNDetector(
                k=sr_cfg.get("k", 50),
                normalize_l2=sr_cfg.get("normalize_l2", True),
                sampling_ratio=alpha,
                seed=seed,
            ).fit(caches[seed]["id_train"])
            id_s = det.score(caches[seed]["id_test"])
            for ood_name in all_ood:
                t0 = time.perf_counter()
                ood_s = det.score(caches[seed]["ood"][ood_name])
                lat_ms = (time.perf_counter() - t0) * 1000.0
                m = compute_ood_metrics(id_s, ood_s)
                m["latency_ms"] = lat_ms
                seed_metrics.append(m)
        agg = summarize_multi_seed(seed_metrics)
        print(
            f"{alpha * 100:6.0f}%    | "
            f"{agg['auroc_mean']:6.2f} ± {agg['auroc_std']:4.2f}%     | "
            f"{agg['fpr95_mean']:6.2f} ± {agg['fpr95_std']:4.2f}%     | "
            f"{agg['latency_ms_mean']:6.2f} ± {agg['latency_ms_std']:4.2f} ms"
        )


def main():
    parser = argparse.ArgumentParser(description="Run Deep kNN Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/ablation_knn.yaml")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()
    run_ablation(args.config, cache_dir_override=args.cache_dir)


if __name__ == "__main__":
    main()
