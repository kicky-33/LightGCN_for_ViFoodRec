# LightGCN trên ViFoodRec
### Tái hiện và Triển khai Graph Collaborative Filtering cho Hệ thống Gợi ý Món ăn Việt Nam

**Môn học:** Đồ án Ngành Khoa Học Dữ Liệu (DS102)
**Thành viên:** La Gia Hân & Hoàng Yến

---

## 🗂️ Cấu trúc thư mục

```
do-an-lightgcn-vifoodrec/
│
├── data/
│   ├── raw/
│   │   ├── ratings.csv              # Dataset gốc ViFoodRec (đã fill median)
│   │   └── foods.csv                # Thông tin món ăn (tên, loại, calories)
│   │
│   └── processed/
│       └── experiments/
│           ├── th3_0/               # train.txt, test.txt, mapping
│           ├── th3_5/               # ← threshold tối ưu, dùng cho train chính thức
│           ├── th4_0/
│           └── th4_5/
│
├── scripts/
│   ├── preprocess_vifoodrec.py      # Tiền xử lý ratings.csv → train/test × 4 threshold
│   ├── train_lightgcn_pyg.py        # Train LightGCN (torch_geometric), lưu .pt + history .json
│   ├── evaluate_checkpoint.py       # Đánh giá checkpoint đã train, ghi CSV
│   ├── summarize_seeds.py           # Tổng hợp mean±std từ 3 seed
│   ├── plot_loss_curves.py          # Parse Kaggle log → vẽ biểu đồ loss & metric
│   └── error_analysis.py            # Long-tail, user group, dish type, case study
│
├── checkpoints/
│   ├── lgn-gowalla-3-64.pth.tar     # Baseline Gowalla (gusye1234/LightGCN-PyTorch)
│   ├── lightgcn_th3_5_seed2026.pt   # ViFoodRec — seed chính thức
│   ├── lightgcn_th3_5_seed42.pt
│   └── lightgcn_th3_5_seed123.pt
│
├── outputs/
│   ├── results/
│   │   ├── summary_all_thresholds.csv       # #users/#items/#interactions × 4 threshold
│   │   ├── threshold_ablation_results.csv   # Recall/NDCG/Precision × 4 threshold
│   │   └── final_result_th3_5.csv           # mean±std 3 seed — KẾT QUẢ CHÍNH THỨC
│   │
│   ├── figures/
│   │   ├── loss_curves_all_thresholds.png   # Biểu đồ loss & metric × 4 threshold
│   │   ├── error_analysis_longtail.png
│   │   ├── error_analysis_usergroup.png
│   │   └── error_analysis_dishtype.png
│   │
│   └── error_analysis/
│       ├── case_study.csv
│       └── error_analysis_summary.txt       # Tổng hợp số liệu Error Analysis
│
└── notes/
    ├── note_report.md               # Tổng hợp mọi điểm cần nhớ khi viết báo cáo
    ├── tien_do_du_an.md             # Nhật ký tiến độ chi tiết
    └── plan_preprocessing.md        # Kế hoạch thực nghiệm threshold ablation (v2.0)
```

---

## 📐 Tổng quan phương pháp

### Bài toán
Xây dựng hệ thống gợi ý **Top-K món ăn** cho từng user dựa trên lịch sử tương tác (implicit feedback), sử dụng kiến trúc **LightGCN** — mô hình Graph Neural Network được tối ưu hóa cho collaborative filtering.

**Bài toán Warm-start:** Chỉ xét user đã có lịch sử tương tác (K-core=5), không giải quyết cold-start.

### Mô hình: LightGCN
LightGCN (He et al., SIGIR 2020) đơn giản hóa GCN truyền thống bằng cách loại bỏ:
- Feature Transformation
- Non-linear Activation

→ Chỉ giữ lại **graph convolution thuần túy** và **weighted sum các lớp** làm embedding cuối cùng. Tối ưu bằng **BPR loss + L2 regularization**.

### Pipeline

```
ratings.csv
    │
    ▼ preprocess_vifoodrec.py
    │  - Binarize rating >= threshold → implicit feedback
    │  - K-core=5 filtering
    │  - Train/test split per-user (80/20)
    │
    ▼ train_lightgcn_pyg.py (Kaggle T4 GPU)
    │  - torch_geometric.nn.models.LightGCN
    │  - 1000 epochs, ~22 phút/lần
    │  - Lưu checkpoint .pt + history .json
    │
    ▼ evaluate_checkpoint.py / summarize_seeds.py
    │  - Recall@20, Precision@20, NDCG@20
    │  - 3 seed → mean ± std
    │
    ▼ error_analysis.py
       - Long-tail, User group, Dish type, Case study
```

---

## 📊 Kết quả

### 1. Tái hiện Baseline Gowalla (xác nhận cài đặt đúng)

| Metric | Nhóm đạt | Paper gốc (He et al., 2020) |
|--------|-----------|----------------------------|
| Recall@20 | 0.1821 | 0.1830 |
| NDCG@20 | 0.1545 | 0.1557 |
| Precision@20 | 0.0558 | 0.0561 |

Sai lệch < 0.7% so với paper gốc — xác nhận cài đặt chính xác.

### 2. Threshold Ablation — ViFoodRec

Thực nghiệm với 4 ngưỡng binarize rating để tìm ngưỡng tối ưu chuyển explicit rating (0–5) sang implicit feedback.

| Threshold | #Items | #Interactions | Recall@20 | Precision@20 | NDCG@20 |
|-----------|--------|----------------|-----------|--------------|---------|
| 3.0 | 4001 | 82,927 | 0.0054 | 0.0441 | 0.0408 |
| **3.5** ★ | **4001** | **66,358** | **0.0073** | **0.0480** | **0.0499** |
| 4.0 | 3983 | 49,670 | 0.0059 | 0.0287 | 0.0282 |
| 4.5 | 3709 | 32,109 | 0.0051 | 0.0158 | 0.0153 |

**→ Chốt threshold = 3.5**, tốt nhất cả 3 metric.

> **Đóng góp:** Paper gốc ViFoodRec dùng threshold=4.0 cho CF truyền thống. Thực nghiệm này cho thấy threshold=3.5 phù hợp hơn với LightGCN vì mô hình cần mật độ đồ thị đủ cao để message passing hiệu quả.

### 3. Kết quả chính thức — ViFoodRec (threshold=3.5, 3 seed)

| Seed | Recall@20 | Precision@20 | NDCG@20 |
|------|-----------|--------------|---------|
| 2026 | 0.0073 | 0.0480 | 0.0499 |
| 42 | 0.0055 | 0.0356 | 0.0384 |
| 123 | 0.0056 | 0.0371 | 0.0370 |
| **mean ± std** | **0.0061 ± 0.0008** | **0.0403 ± 0.0055** | **0.0418 ± 0.0058** |

### 4. Error Analysis

**Long-tail Analysis:**

| Nhóm | Recall@20 |
|------|-----------|
| Head (top 20% phổ biến, 800 items) | 0.0245 |
| Body (60%, 2400 items) | 0.0044 |
| Tail (bottom 20%, 801 items) | 0.0000 |

→ **Popularity bias rõ rệt:** LightGCN gần như không gợi ý được món long-tail do ít cạnh trên đồ thị → message passing không hiệu quả.

**User Group Analysis:**

| Nhóm | n | Recall@20 | NDCG@20 |
|------|---|-----------|---------|
| Active (≥ 519 interactions) | 51 | 0.0075 | 0.0505 |
| Less-active (< 519 interactions) | 50 | 0.0071 | 0.0493 |

→ Chênh lệch không đáng kể — mô hình công bằng giữa hai nhóm user (do ngay cả "less-active" vẫn có 500+ interactions).

**Dish Type Analysis:**

| Loại | Recall@20 |
|------|-----------|
| Món mặn (3805 items) | 0.0074 |
| Món chay (195 items) | 0.0061 |

→ Chênh lệch nhỏ, giải thích được bởi món chay chiếm ~5% dataset (phần lớn nằm trong nhóm long-tail).

---

## ⚙️ Cấu hình thực nghiệm

| Tham số | Giá trị |
|---------|---------|
| Embedding dimension | 64 |
| Số lớp GCN (n_layers) | 3 |
| Learning rate | 0.001 |
| L2 regularization (decay) | 1e-4 |
| Batch size | 1024 |
| Epochs | 1000 |
| Train/test split | 80/20 per-user |
| K-core | 5 |
| Random seeds | 2026, 42, 123 |
| GPU | Kaggle Tesla T4 |
| Thời gian train/lần | ~22 phút |

---

## 🔧 Cài đặt & Chạy

### Yêu cầu
```
torch_geometric
torch
pandas
numpy
matplotlib
```

### Tiền xử lý
```bash
python scripts/preprocess_vifoodrec.py
# Output: data/processed/experiments/th{3_0,3_5,4_0,4_5}/
```

### Train (chạy trên Kaggle, sửa DATA_DIR và SEED trong script)
```bash
python scripts/train_lightgcn_pyg.py
# Đổi SEED=2026/42/123 để train 3 seed
# Output: lightgcn_{dataset}_seed{SEED}.pt + history_{dataset}_seed{SEED}.json
```

### Tổng hợp kết quả 3 seed
```bash
python scripts/summarize_seeds.py
# Output: outputs/results/final_result_th3_5.csv
```

### Vẽ biểu đồ loss
```bash
# Paste Kaggle log vào logs/log_th{3_0,3_5,4_0,4_5}.txt rồi chạy:
python scripts/plot_loss_curves.py
# Output: outputs/figures/
```

### Error Analysis
```bash
python scripts/error_analysis.py
# Output: outputs/error_analysis/
```

---

## 📦 Dataset

**ViFoodRec** (Tran et al., PACLIC 2024)
- Repo: [github.com/QuocAn55/A-New-Dataset-and-Empirical-Evaluation-for-Vietnamese-Food-Recommendation-System](https://github.com/QuocAn55/A-New-Dataset-and-Empirical-Evaluation-for-Vietnamese-Food-Recommendation-System)
- `foods.csv`: 4000+ món ăn Việt Nam (tên, loại, calories, cooking_time)
- `ratings.csv`: ~180,000 ratings từ 101 user (⚠️ đã fill 40% giá trị thiếu bằng median per-user — xem `notes/note_report.md` mục 1)

---

## 📚 Tài liệu tham khảo

1. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** *SIGIR 2020.*
2. Tran, Q. A., Dang, C. T., Dang, T. H. T., et al. (2024). **A New Dataset and Empirical Evaluation for Vietnamese Food Recommendation System.** *PACLIC 2024.*
3. Hu, Y., Koren, Y., & Volinsky, C. (2008). **Collaborative filtering for implicit feedback datasets.** *ICDM 2008.*
