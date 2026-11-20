#!/usr/bin/env python3
"""
Script 1: Huấn luyện Backbone ResNet-18 trên CIFAR-10 (qua 3 random seeds)
và trích xuất Feature Cache (.npz: features 512d + logits 10d) cho toàn bộ tập ID & OOD.

Cách chạy trên Google Colab Pro GPU:
    python scripts/train_and_extract.py --config configs/baseline_cifar10.yaml

Cách tạo dữ liệu giả lập (Smoke Test 2 giây trên máy cá nhân không cần GPU):
    python scripts/train_and_extract.py --synthetic
"""
import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.datasets import (
    generate_synthetic_cache,
    get_dataloader,
    save_feature_cache,
)
from src.utils.seed import get_hardware_info, set_seed


def extract_split(model, dataloader, device):
    """Chạy forward pass qua 1 tập dữ liệu để trích xuất features (N x 512) và logits (N x 10)."""
    import torch

    model.eval()
    all_feats, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            logits, feats = model.forward_features(images)
            all_feats.append(feats.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_feats, axis=0),
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def train_and_extract_real(config_path: str, epochs_override: int = None, cache_dir_override: str = None):
    import torch
    import torch.nn as nn
    import yaml
    from src.models.backbone import build_backbone, count_parameters

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hw = get_hardware_info()
    device = torch.device(hw["device"])
    print(f"[Hardware Info] Device: {hw['device']} | GPU: {hw['gpu_name']} | OS: {hw['os']}")

    seeds = cfg.get("seeds", [0, 1, 2])
    ds_cfg = cfg["dataset"]
    bb_cfg = cfg["backbone"]

    data_root = ds_cfg.get("data_root", "./data/raw")
    cache_dir = cache_dir_override or ds_cfg.get("cache_dir", "./data/extracted_cache")
    epochs = epochs_override if epochs_override is not None else bb_cfg.get("epochs", 50)
    batch_size = bb_cfg.get("batch_size", 128)

    # Tải DataLoader cho ID Train & ID Test
    train_loader_eval = get_dataloader("cifar10", data_root=data_root, split="train", batch_size=batch_size)
    train_loader = get_dataloader("cifar10", data_root=data_root, split="train", batch_size=batch_size)
    id_test_loader = get_dataloader("cifar10", data_root=data_root, split="test", batch_size=batch_size)

    ood_names = ds_cfg.get("near_ood", []) + ds_cfg.get("far_ood", [])
    ood_loaders = {
        name: get_dataloader(name, data_root=data_root, split="test", batch_size=batch_size)
        for name in ood_names
    }

    os.makedirs("./checkpoints", exist_ok=True)

    for seed in seeds:
        print(f"\n==================== SEED {seed} ====================")
        set_seed(seed)
        model = build_backbone(bb_cfg.get("arch", "resnet18_cifar"), num_classes=ds_cfg.get("num_classes", 10))
        model = model.to(device)
        if seed == seeds[0]:
            print(f"[Model Info] Backbone: {bb_cfg.get('arch')} | Trainable Params: {count_parameters(model):,}")

        ckpt_path = os.path.join("./checkpoints", f"resnet18_cifar10_seed{seed}.pth")
        if os.path.exists(ckpt_path):
            print(f"[Checkpoint] Loading existing checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=bb_cfg.get("lr", 0.1),
                momentum=bb_cfg.get("momentum", 0.9),
                weight_decay=bb_cfg.get("weight_decay", 5e-4),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

            t0 = time.time()
            for ep in range(1, epochs + 1):
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += images.size(0)
                scheduler.step()
                if ep == 1 or ep % 10 == 0 or ep == epochs:
                    print(
                        f"  Epoch [{ep:02d}/{epochs:02d}] "
                        f"Loss: {running_loss / total:.4f} | Train Acc: {100.0 * correct / total:.2f}%"
                    )
            print(f"[Training Finished] Time elapsed: {time.time() - t0:.1f}s")
            torch.save(model.state_dict(), ckpt_path)

        # Trích xuất Feature Cache cho ID Train (Feature Bank của kNN) & ID Test
        print(f"[Extracting] CIFAR-10 Train (Feature Bank) & Test...")
        tr_f, tr_l, tr_y = extract_split(model, train_loader_eval, device)
        save_feature_cache(cache_dir, seed, "cifar10_train", tr_f, tr_l, tr_y)

        te_f, te_l, te_y = extract_split(model, id_test_loader, device)
        id_acc = float(np.mean(np.argmax(te_l, axis=1) == te_y) * 100.0)
        print(f"  -> CIFAR-10 ID Test Accuracy (Seed {seed}): {id_acc:.2f}%")
        save_feature_cache(cache_dir, seed, "cifar10_test", te_f, te_l, te_y)

        # Trích xuất cho toàn bộ các tập Near-OOD và Far-OOD
        for ood_name, loader in ood_loaders.items():
            print(f"[Extracting] OOD Dataset: {ood_name}...")
            of, ol, oy = extract_split(model, loader, device)
            save_feature_cache(cache_dir, seed, ood_name, of, ol, oy)

    print(f"\n[Done] Toàn bộ Feature Cache (.npz) đã được lưu tại: {os.path.abspath(cache_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Train Backbone & Extract OOD Feature Caches (.npz)")
    parser.add_argument("--config", type=str, default="configs/baseline_cifar10.yaml", help="Đường dẫn file YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epochs huấn luyện")
    parser.add_argument("--cache-dir", type=str, default=None, help="Thư mục lưu file .npz (ví dụ Google Drive)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Sinh nhanh Feature Cache giả lập (3 seeds) để kiểm thử toàn bộ code trong 2 giây",
    )
    args = parser.parse_args()

    if args.synthetic:
        cache_dir = args.cache_dir or "./data/extracted_cache"
        generate_synthetic_cache(cache_dir=cache_dir, seeds=(0, 1, 2))
        print(f"[Synthetic Cache Ready] Đã tạo bộ cache thử nghiệm (seeds 0, 1, 2) tại: {os.path.abspath(cache_dir)}")
    else:
        train_and_extract_real(args.config, epochs_override=args.epochs, cache_dir_override=args.cache_dir)


if __name__ == "__main__":
    main()
