import os
from typing import Dict, Tuple
import numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(image_size: int = 32, is_train: bool = False, is_grayscale: bool = False):
    """
    Khởi tạo chuỗi phép biến đổi ảnh (Transforms) chuẩn cho bài toán OOD.
    QUAN TRỌNG: Mọi tập OOD đều phải chuẩn hóa bằng đúng mean và std của tập Train ID (CIFAR-10).
    """
    from torchvision import transforms

    transform_list = []
    if is_grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    if is_train:
        transform_list.extend([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return transforms.Compose(transform_list)


def get_dataloader(
    dataset_name: str,
    data_root: str = "./data/raw",
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 32,
):
    """
    Tải tự động các bộ dữ liệu ID (CIFAR-10) và Near/Far-OOD (CIFAR-100, SVHN, MNIST, DTD)
    qua torchvision.datasets.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets

    name = dataset_name.lower()
    is_train = split == "train"
    os.makedirs(data_root, exist_ok=True)

    if name == "cifar10":
        tf = get_transforms(image_size=image_size, is_train=is_train, is_grayscale=False)
        ds = datasets.CIFAR10(root=data_root, train=is_train, download=True, transform=tf)
    elif name == "cifar100":
        tf = get_transforms(image_size=image_size, is_train=False, is_grayscale=False)
        ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=tf)
    elif name == "svhn":
        tf = get_transforms(image_size=image_size, is_train=False, is_grayscale=False)
        ds = datasets.SVHN(root=data_root, split="test", download=True, transform=tf)
    elif name == "mnist":
        tf = get_transforms(image_size=image_size, is_train=False, is_grayscale=True)
        ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tf)
    elif name in ("dtd", "textures"):
        tf = get_transforms(image_size=image_size, is_train=False, is_grayscale=False)
        ds = datasets.DTD(root=data_root, split="test", download=True, transform=tf)
    else:
        raise ValueError(f"Dataset '{dataset_name}' chưa được hỗ trợ. Chọn trong: cifar10, cifar100, svhn, mnist, dtd.")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )


def save_feature_cache(
    cache_dir: str,
    seed: int,
    dataset_name: str,
    features: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
) -> str:
    """
    Lưu đặc trưng tầng áp chót (features: N x 512) và điểm số phân lớp (logits: N x 10)
    ra file nén .npz để cả nhóm (MSP, Energy, kNN) dùng chung mà không cần chạy lại GPU.
    """
    seed_dir = os.path.join(cache_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    file_path = os.path.join(seed_dir, f"{dataset_name}.npz")
    np.savez_compressed(
        file_path,
        features=features.astype(np.float32),
        logits=logits.astype(np.float32),
        labels=labels.astype(np.int64),
    )
    return file_path


def load_feature_cache(cache_dir: str, seed: int, dataset_name: str) -> Dict[str, np.ndarray]:
    """Đọc file .npz chứa features (cho kNN) và logits (cho MSP, Energy)."""
    file_path = os.path.join(cache_dir, f"seed_{seed}", f"{dataset_name}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy cache tại '{file_path}'. "
            f"Hãy chạy 'python scripts/train_and_extract.py' (hoặc dùng cờ --synthetic để test pipeline)."
        )
    data = np.load(file_path)
    return {
        "features": data["features"],
        "logits": data["logits"],
        "labels": data["labels"],
    }


def generate_synthetic_cache(
    cache_dir: str = "./data/extracted_cache",
    seeds: Tuple[int, ...] = (0, 1, 2),
    feature_dim: int = 512,
    num_classes: int = 10,
) -> None:
    """
    Sinh bộ Feature Cache giả lập mô phỏng đặc trưng tầng áp chót của ResNet-18:
      - Mỗi lớp ID kích hoạt mạnh một nhóm kênh đặc trưng đặc thù (class-specific feature channels).
      - Near-OOD kích hoạt chồng lấn nhiều lớp kèm nhiễu nền.
      - Far-OOD có phân phối kênh phẳng/nhiễu và biến thiên mạnh về độ lớn norm (norm shift).
    """
    scale = 1.0 / np.sqrt(feature_dim)
    block = max(1, feature_dim // num_classes)

    for seed in seeds:
        rng = np.random.default_rng(seed + 100)
        # Mỗi tâm lớp c kích hoạt mạnh trên khối kênh [c*block : (c+1)*block]
        centers = np.zeros((num_classes, feature_dim), dtype=np.float32)
        for c in range(num_classes):
            start = c * block
            end = feature_dim if c == num_classes - 1 else (c + 1) * block
            centers[c, start:end] = rng.uniform(0.7, 1.3, size=(end - start,))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        centers *= 4.0

        # Ma trận trọng số tầng phân lớp cuối W (feature_dim x num_classes)
        W = (centers / 4.0).T * 3.5

        def _make_split(n_samples: int, mode: str):
            y = rng.integers(0, num_classes, size=n_samples)
            if mode == "id":
                # ID: Tập trung sát tâm cụm của từng lớp
                feats = centers[y] + rng.normal(0, 1.1 * scale, size=(n_samples, feature_dim)) * 4.0
                feats *= rng.uniform(0.85, 1.15, size=(n_samples, 1))
            elif mode == "near_ood":
                # Near-OOD (CIFAR-100): Nằm ở vùng trung gian giữa 2 lớp kèm nhiễu
                c1 = centers[y]
                c2 = centers[(y + 1) % num_classes]
                feats = 0.45 * c1 + 0.45 * c2 + rng.normal(0, 2.1 * scale, size=(n_samples, feature_dim)) * 4.0
            elif mode == "far_ood_norm_shift":
                # Far-OOD có norm nhỏ (làm khoảng cách Euclid chưa chuẩn hóa L2 tới gốc tọa độ nhỏ giả tạo)
                # nhưng hướng vector trên mặt cầu đơn vị (sau L2-norm) lại cách xa cả 10 cụm ID
                feats = np.abs(rng.normal(0, 1.0, size=(n_samples, feature_dim)))
                feats /= np.linalg.norm(feats, axis=1, keepdims=True)
                feats *= rng.uniform(1.8, 2.4, size=(n_samples, 1))
            else:
                # Far-OOD chuẩn
                feats = np.abs(rng.normal(0, 1.0, size=(n_samples, feature_dim)))
                feats /= np.linalg.norm(feats, axis=1, keepdims=True)
                feats *= rng.uniform(2.5, 3.5, size=(n_samples, 1))

            feats = np.maximum(feats, 0.0).astype(np.float32)
            logits = (feats @ W).astype(np.float32)
            return feats, logits, y

        id_tr_f, id_tr_l, id_tr_y = _make_split(5000, "id")
        save_feature_cache(cache_dir, seed, "cifar10_train", id_tr_f, id_tr_l, id_tr_y)

        id_te_f, id_te_l, id_te_y = _make_split(1000, "id")
        save_feature_cache(cache_dir, seed, "cifar10_test", id_te_f, id_te_l, id_te_y)

        near_f, near_l, near_y = _make_split(1000, "near_ood")
        save_feature_cache(cache_dir, seed, "cifar100", near_f, near_l, near_y)

        svhn_f, svhn_l, svhn_y = _make_split(1000, "far_ood_norm_shift")
        save_feature_cache(cache_dir, seed, "svhn", svhn_f, svhn_l, svhn_y)

        mnist_f, mnist_l, mnist_y = _make_split(1000, "far_ood")
        save_feature_cache(cache_dir, seed, "mnist", mnist_f, mnist_l, mnist_y)

        dtd_f, dtd_l, dtd_y = _make_split(1000, "far_ood_norm_shift")
        save_feature_cache(cache_dir, seed, "dtd", dtd_f, dtd_l, dtd_y)
