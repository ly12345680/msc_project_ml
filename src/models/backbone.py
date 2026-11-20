from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """
    Kiến trúc ResNet-18 tối ưu cho ảnh 32x32 (CIFAR-10/100) theo chuẩn OpenOOD:
      - Thay conv1 7x7 (stride 2) bằng conv1 3x3 (stride 1, padding 1).
      - Bỏ tầng MaxPool đầu tiên để giữ độ phân giải không gian cho ảnh 32x32.
      - Trả về đồng thời vector đặc trưng tầng áp chót (z in R^512 cho Deep kNN)
        và vector điểm số phân lớp (logits in R^10 cho MSP và Energy).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.feature_dim = base.fc.in_features  # 512
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (logits [B, C], penultimate_features [B, 512])."""
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        return logits, feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_features(x)
        return logits


def build_backbone(arch: str = "resnet18_cifar", num_classes: int = 10) -> nn.Module:
    if arch == "resnet18_cifar":
        return ResNet18CIFAR(num_classes=num_classes)
    raise ValueError(f"Kiến trúc '{arch}' chưa được hỗ trợ.")


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số huấn luyện của mô hình (phục vụ báo cáo Mục 4.2)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
