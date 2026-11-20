#!/usr/bin/env python3
"""
Script 2: Chạy Benchmark so sánh 3 phương pháp Post-hoc (MSP vs. Energy vs. Deep kNN)
trên toàn bộ các tập Near-OOD và Far-OOD qua 3 random seeds (báo cáo mean ± std).

Cách chạy:
    python scripts/run_benchmark.py --config configs/baseline_cifar10.yaml
"""
import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import load_feature_cache
from src.methods import DeepKNNDetector, EnergyDetector, MSPDetector
from src.utils.metrics import compute_ood_metrics, summarize_multi_seed
from src.utils.seed import get_hardware_info


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        return {
            "seeds": [0, 1, 2],
            "dataset": {
                "id_name": "cifar10",
                "near_ood": ["cifar100"],
                "far_ood": ["svhn", "mnist", "dtd"],
                "cache_dir": "./data/extracted_cache",
            },
            "methods": {
                "msp": {"enabled": True},
                "energy": {"enabled": True, "temperature": 1.0},
                "knn": {"enabled": True, "k": 50, "normalize_l2": True, "sampling_ratio": 1.0},
            },
            "output": {"results_dir": "./results"},
        }


def run_benchmark(config_path: str, cache_dir_override: str = None):
    cfg = load_config(config_path)
    seeds: List[int] = cfg.get("seeds", [0, 1, 2])
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["methods"]

    cache_dir = cache_dir_override or ds_cfg.get("cache_dir", "./data/extracted_cache")
    results_dir = cfg.get("output", {}).get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    near_ood_list: List[str] = ds_cfg.get("near_ood", ["cifar100"])
    far_ood_list: List[str] = ds_cfg.get("far_ood", ["svhn", "mnist", "dtd"])
    all_ood = [(name, "Near-OOD") for name in near_ood_list] + [(name, "Far-OOD") for name in far_ood_list]

    hw = get_hardware_info()
    print("================================================================================")
    print("   CHỦ ĐỀ 11: OUT-OF-DISTRIBUTION DETECTION BENCHMARK (MSP vs Energy vs kNN)")
    print("================================================================================")
    print(f"Hardware : {hw['device'].upper()} ({hw['gpu_name']}) | OS: {hw['os']}")
    print(f"Seeds    : {seeds} | Cache Dir: {cache_dir}")
    print("--------------------------------------------------------------------------------")

    # Khởi tạo danh sách detector cần đánh giá
    def build_detectors(seed: int):
        detectors = []
        if m_cfg.get("msp", {}).get("enabled", True):
            detectors.append(MSPDetector())
        if m_cfg.get("energy", {}).get("enabled", True):
            t = m_cfg["energy"].get("temperature", 1.0)
            detectors.append(EnergyDetector(temperature=t))
        if m_cfg.get("knn", {}).get("enabled", True):
            k_cfg = m_cfg["knn"]
            detectors.append(
                DeepKNNDetector(
                    k=k_cfg.get("k", 50),
                    normalize_l2=k_cfg.get("normalize_l2", True),
                    sampling_ratio=k_cfg.get("sampling_ratio", 1.0),
                    seed=seed,
                )
            )
        return detectors

    # Lưu kết quả theo cấu trúc: summary_records[(method_name, ood_name, ood_group)] = [seed0_dict, ...]
    raw_records: Dict[tuple, List[Dict[str, float]]] = {}

    for seed in seeds:
        id_train = load_feature_cache(cache_dir, seed, "cifar10_train")
        id_test = load_feature_cache(cache_dir, seed, "cifar10_test")

        detectors = build_detectors(seed)
        for det in detectors:
            det.fit(id_train)
            id_scores = det.score(id_test)

            for ood_name, ood_group in all_ood:
                ood_test = load_feature_cache(cache_dir, seed, ood_name)
                t0 = time.perf_counter()
                ood_scores = det.score(ood_test)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                metrics = compute_ood_metrics(id_scores, ood_scores)
                metrics["latency_ms"] = latency_ms

                key = (det.name, ood_name, ood_group)
                raw_records.setdefault(key, []).append(metrics)

    # Tổng hợp mean ± std qua các seeds
    summary_rows = []
    print(f"\n{'Method':<12} | {'OOD Dataset':<12} | {'Group':<10} | {'AUROC (↑ %)':<18} | {'FPR@95 (↓ %)':<18} | {'Latency (ms)':<12}")
    print("-" * 94)

    for (method_name, ood_name, ood_group), seed_list in raw_records.items():
        agg = summarize_multi_seed(seed_list)
        auroc_str = f"{agg['auroc_mean']:.2f} ± {agg['auroc_std']:.2f}"
        fpr95_str = f"{agg['fpr95_mean']:.2f} ± {agg['fpr95_std']:.2f}"
        lat_str = f"{agg['latency_ms_mean']:.2f}"
        print(f"{method_name:<12} | {ood_name:<12} | {ood_group:<10} | {auroc_str:<18} | {fpr95_str:<18} | {lat_str:<12}")

        summary_rows.append({
            "method": method_name,
            "ood_dataset": ood_name,
            "ood_group": ood_group,
            **agg,
        })

    # Lưu CSV
    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "ood_dataset",
                "ood_group",
                "auroc_mean",
                "auroc_std",
                "fpr95_mean",
                "fpr95_std",
                "latency_ms_mean",
                "latency_ms_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("-" * 94)
    print(f"[Saved] Bảng kết quả tổng hợp đã lưu tại: {os.path.abspath(csv_path)}")


def main():
    parser = argparse.ArgumentParser(description="Run Post-hoc OOD Benchmark (MSP, Energy, Deep kNN)")
    parser.add_argument("--config", type=str, default="configs/baseline_cifar10.yaml")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args.config, cache_dir_override=args.cache_dir)


if __name__ == "__main__":
    main()
