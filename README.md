# Chủ đề 11: Out-of-Distribution Detection (OOD)

**Môn học:** Machine Learning and Applications (Chương trình Thạc sĩ Khoa học Máy tính)  
**Nhóm chủ đề:** Nhóm IV – OOD / NCD / GCD (Độ khó: Khá)  
**Tài liệu báo cáo & đề cương:** [`Chu_de_11_OOD_Dan_Y_va_Du_Lieu.docx`](./Chu_de_11_OOD_Dan_Y_va_Du_Lieu.docx)

---

## 1. Mục tiêu & Phương pháp so sánh

Dự án nghiên cứu và tái hiện 3 phương pháp **Post-hoc Out-of-Distribution (OOD) Detection** đại diện cho 3 tầng biểu diễn khác nhau của mạng Deep Neural Network (`ResNet-18` huấn luyện trên `CIFAR-10` qua 3 random seeds `0, 1, 2`):

1. **MSP (Maximum Softmax Probability – Baseline):** [Hendrycks & Gimpel, ICLR 2017](https://arxiv.org/abs/1610.02136) — dựa trên xác suất Softmax cực đại ở tầng đầu ra (`src/methods/msp.py`).
2. **Energy-based OOD Detection:** [Liu et al., NeurIPS 2020](https://arxiv.org/abs/2010.03759) — dựa trên hàm năng lượng tự do Helmholtz tính trên vector Logits trước Softmax (`src/methods/energy.py`).
3. **Deep k-Nearest Neighbors (Deep kNN – Nhánh trọng tâm):** [Sun et al., ICML 2022](https://arxiv.org/abs/2204.06507) — phương pháp khoảng cách phi tham số (*non-parametric*) trên vector đặc trưng tầng áp chót đã chuẩn hóa $L_2$ (`src/methods/knn.py`).

**Độ đo đánh giá chuẩn:**
* **AUROC ($\uparrow$ %):** Diện tích dưới đường cong ROC (càng cao càng tốt).
* **FPR@95 ($\downarrow$ %):** Tỷ lệ mẫu OOD bị nhận nhầm là ID khi ngưỡng quyết định $\lambda$ được cố định tại $\text{TPR}_{\text{ID}} = 95\%$ (càng thấp càng tốt).

---

## 2. Cấu trúc thư mục dự án (Đáp ứng Mục 4.3 & Rubric 10% Mã nguồn)

```text
msc_project_ml/
├── README.md                                 # Hướng dẫn cài đặt và tái lập toàn bộ kết quả
├── requirements.txt                          # Danh sách thư viện Python 3.10+
├── Chu_de_11_OOD_Dan_Y_va_Du_Lieu.docx       # File Word Đề cương & Nội dung chi tiết phần Dữ liệu
├── configs/                                  # File cấu hình YAML cho từng thí nghiệm
│   ├── baseline_cifar10.yaml                 # Cấu hình benchmark chính (MSP vs Energy vs kNN, 3 seeds)
│   └── ablation_knn.yaml                     # Cấu hình 3 thí nghiệm Ablation cho nhánh Deep kNN
├── src/
│   ├── data/
│   │   └── datasets.py                       # Tải CIFAR-10 (ID), CIFAR-100, SVHN, MNIST, DTD (OOD) & Cache .npz
│   ├── models/
│   │   └── backbone.py                       # Kiến trúc ResNet-18 (32x32 CIFAR) trích đồng thời logits & features
│   ├── methods/                              # Mỗi thành viên phụ trách độc lập 1 module phương pháp
│   │   ├── base.py                           # Giao diện BaseOODDetector chuẩn
│   │   ├── msp.py                            # Thành viên 1: Maximum Softmax Probability
│   │   ├── energy.py                         # Thành viên 2: Energy-based OOD
│   │   └── knn.py                            # Thành viên 3 (Bạn): Deep kNN (L2-norm, Feature Bank, FAISS/NumPy)
│   └── utils/
│       ├── metrics.py                        # Tính chính xác AUROC, FPR@95 và tổng hợp mean ± std
│       └── seed.py                           # Cố định random seed & ghi cấu hình phần cứng
├── scripts/
│   ├── train_and_extract.py                  # Huấn luyện ResNet-18 & xuất Feature Cache (.npz)
│   ├── run_benchmark.py                      # Chạy so sánh MSP vs Energy vs Deep kNN qua 3 seeds
│   └── run_ablation_knn.py                   # Chạy 3 Ablation Studies cho Deep kNN (k-sweep, L2-norm, alpha)
├── notebooks/
│   └── colab_gpu_runner.ipynb                # Notebook chạy trên Google Colab Pro GPU + Shared Google Drive
└── tests/
    └── test_ood_pipeline.py                  # Unit test kiểm tra toàn bộ thuật toán & độ đo
```

---

## 3. Hướng dẫn cài đặt & Tái lập kết quả

### Bước 1: Cài đặt môi trường
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bước 2: Chạy kiểm thử nhanh toàn bộ Pipeline (Smoke Test trong 2 giây, không cần GPU)
Nếu muốn kiểm tra ngay mã nguồn `MSP`, `Energy`, `Deep kNN` và `Ablation Study` trên máy cá nhân trước khi chạy GPU:
```bash
# 1. Sinh bộ Feature Cache giả lập có cấu trúc hình học thực tế cho 3 seeds (0, 1, 2)
python scripts/train_and_extract.py --synthetic

# 2. Chạy bảng so sánh chính: MSP vs. Energy vs. Deep kNN (xuất CSV vào results/benchmark_summary.csv)
python scripts/run_benchmark.py --config configs/baseline_cifar10.yaml

# 3. Chạy 3 thí nghiệm Ablation Study của nhánh Deep kNN
python scripts/run_ablation_knn.py --config configs/ablation_knn.yaml
```

### Bước 3: Huấn luyện thực tế trên Google Colab Pro GPU & Chia sẻ Feature Cache cho cả nhóm
1. Mở file [`notebooks/colab_gpu_runner.ipynb`](./notebooks/colab_gpu_runner.ipynb) trên **Google Colab Pro** (chọn GPU **L4** hoặc **A100**).
2. Mount thư mục Google Drive dùng chung của nhóm và chạy:
   ```bash
   python scripts/train_and_extract.py --config configs/baseline_cifar10.yaml --cache-dir "/content/drive/MyDrive/MSC_OOD_Topic11_Shared/extracted_cache"
   ```
3. Sau khi các file `.npz` (`cifar10_train.npz`, `cifar10_test.npz`, `cifar100.npz`, `svhn.npz`, `mnist.npz`, `dtd.npz`) được lưu vào Google Drive chung cho cả 3 seeds, mọi thành viên trong nhóm có thể chạy `run_benchmark.py` và `run_ablation_knn.py` chỉ trong vài giây.

---

## 4. Nguồn tham khảo & Trích dẫn
* **Benchmark chuẩn:** [OpenOOD / OODSurvey](https://github.com/Jingkang50/OODSurvey) — Yang et al. (2021), *"Generalized Out-of-Distribution Detection: A Survey"*.
* **Phương pháp Deep kNN:** Sun et al. (ICML 2022), *"Out-of-Distribution Detection with Deep Nearest Neighbors"*, [arXiv:2204.06507](https://arxiv.org/abs/2204.06507).
* **Phương pháp Energy:** Liu et al. (NeurIPS 2020), *"Energy-based Out-of-distribution Detection"*, [arXiv:2010.03759](https://arxiv.org/abs/2010.03759).
* **Phương pháp MSP:** Hendrycks & Gimpel (ICLR 2017), *"A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"*, [arXiv:1610.02136](https://arxiv.org/abs/1610.02136).
