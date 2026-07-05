# LightGCN trên ViFoodRec
### Tái hiện mô hình LightGCN (SIGIR 2020) và đánh giá hiệu năng trên bộ dữ liệu gợi ý món ăn Việt Nam (ViFoodRec - PACLIC 2024).

**Môn học:** Đồ án Ngành Khoa Học Dữ Liệu (DS102)
**Thành viên:** La Gia Hân & Hoàng Yến
**Source code:** [github.com/kicky-33/LightGCN_for_ViFoodRec](https://github.com/kicky-33/LightGCN_for_ViFoodRec)

---

## Cấu trúc thư mục

```
LightGCN_for_ViFoodRec/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── ratings.csv              # Dataset gốc ViFoodRec (đã fill median)
│   │   └── foods.csv                # Thông tin món ăn (tên, loại, calories)
│   ├── gowalla/
│   │   ├── train.txt                # Dataset Gowalla gốc (gusye1234/LightGCN-PyTorch)
│   │   └── test.txt
│   └── processed/
│       └── experiments/
│           ├── th3_0/               # train.txt, test.txt, mapping
│           ├── th3_5/               # ← threshold tối ưu, dùng cho train chính thức
│           ├── th4_0/
│           └── th4_5/
│
├── scripts/
│   ├── preprocess_vifoodrec.py         # Tiền xử lý ratings.csv → train/test × 4 threshold
│   ├── train_lightgcn_pyg.py           # Train LightGCN (torch_geometric), lưu .pt + history .json
│   ├── evaluate_checkpoint.py          # Đánh giá checkpoint LightGCN đã train, ghi CSV
│   ├── evaluate_gowalla_checkpoint.py  # Đánh giá lại checkpoint Gowalla gốc (gusye1234), không train lại
│   ├── summarize_seeds.py              # Tổng hợp mean±std qua nhiều seed
│   ├── plot_loss_curves.py             # Parse Kaggle log → vẽ biểu đồ loss & metric
│   ├── error_analysis.py               # User group, dish type, case study
│   ├── longtail_analysis.py            # Long-tail Head/Body/Tail cho cả 3 model × seed
│   ├── run_all_baselines.py            # Train + đánh giá BPR-MF / NGCF / LightGCN (so sánh chính thức)
│   ├── train.py                        # Train 1 model đơn lẻ qua CLI (--model --threshold --seed)
│   ├── model.py                        # BPRMF, NGCF (dùng cho train.py và run_all_baselines.py)
│   ├── losses.py                       # BPR loss
│   ├── evaluate.py                     # Recall/Precision/NDCG@K (all-ranking)
│   ├── data.py                         # InteractionData: load + build norm adjacency + sampler
│   ├── config.py                       # Cấu hình chung, nhận tham số qua argparse
│   └── vifoodrec_demo.html             # Demo tương tác: gợi ý món ăn + toàn bộ bảng/biểu đồ kết quả
│
├── checkpoints/
│   ├── gowalla/
│   │   └── lgn-gowalla-3-64.pth.tar    # Baseline Gowalla — checkpoint gốc gusye1234/LightGCN-PyTorch
│   ├── lightgcn_th3_5_seed{2026,42,123,...}.pt
│   ├── bprmf_th3_5_seed{2026,42,123,...}.pt
│   └── ngcf_th3_5_seed{2026,42,123,...}.pt
│   # Quy ước đặt tên thống nhất: checkpoints/{model}_th{threshold}_seed{seed}.pt
│
└── outputs/
    ├── results/
    │   ├── summary_all_thresholds.csv       # #users/#items/#interactions × 4 threshold
    │   ├── threshold_ablation_results.csv   # Recall/NDCG/Precision × 4 threshold (3 seed)
    │   ├── final_result_th3_5.csv           # mean±std, 5 seed — KẾT QUẢ CHÍNH THỨC (3 model)
    │   └── gowalla_reproduction.csv         # Kết quả evaluate_gowalla_checkpoint.py
    │
    ├── predictions/
    │   └── {model}_th3_5_seed{seed}.npy     # Top-K prediction, input cho longtail_analysis.py
    │
    ├── figures/
    │   ├── loss_curves_all_thresholds.png   # Biểu đồ loss & metric × 4 threshold
    │   ├── ablation_layers_recall.png       # Biểu đồ Recall@20 theo số lớp K
    │   ├── error_analysis_usergroup.png
    │   └── error_analysis_dishtype.png
    │
    └── error_analysis/
        ├── case_study.csv
        ├── error_analysis_summary.txt
        └── longtail_per_seed.csv            # Ghi bởi longtail_analysis.py (3 model × seed)
```

---

## Tổng quan phương pháp

### Bài toán
Xây dựng hệ thống gợi ý **Top-K món ăn** cho từng user dựa trên lịch sử tương tác (implicit feedback), sử dụng kiến trúc **LightGCN** — mô hình Graph Neural Network được tối ưu hóa cho collaborative filtering, so sánh với hai baseline **BPR-MF** và **NGCF**.

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
    │  - Train/test split per-user (80/20, stratified theo số tương tác)
    │
    ▼ train_lightgcn_pyg.py (LightGCN)  /  run_all_baselines.py (BPR-MF, NGCF, LightGCN)
    │  - batch_size=1024 dùng chung cho cả 3 model
    │  - Tối đa 1000 epochs, ~22 phút/lần chạy (LightGCN, Kaggle T4)
    │  - Lưu checkpoint checkpoints/{model}_th{threshold}_seed{seed}.pt
    │
    ▼ evaluate_checkpoint.py / evaluate_gowalla_checkpoint.py / summarize_seeds.py
    │  - Recall@20, Precision@20, NDCG@20 (all-ranking protocol)
    │  - Ablation threshold & số lớp K: 3 seed → mean ± std
    │  - Kết quả chính thức (so sánh 3 model): 5 seed → mean ± std
    │
    ▼ error_analysis.py + longtail_analysis.py
       - Long-tail (3 model), User group (top/bottom 25%, 3 model), Dish type, Case study
```

---

## Kết quả

### 1. Tái hiện Baseline Gowalla (xác nhận cài đặt đúng)

| Metric | Nhóm đạt | Paper gốc (He et al., 2020) | Sai lệch |
|--------|-----------|----------------------------|----------|
| Recall@20 | 0.1821 | 0.1830 | −0.49% |
| NDCG@20 | 0.1545 | 0.1557 | −0.77% |
| Precision@20 | 0.0558 | 0.0561 | −0.53% |

Sai lệch **≤ 0.77%** so với paper gốc trên cả 3 metric — xác nhận cài đặt pipeline chính xác (RQ1). Có thể kiểm tra lại không cần train:
```bash
python scripts/evaluate_gowalla_checkpoint.py --ckpt checkpoints/gowalla/lgn-gowalla-3-64.pth.tar --data-dir data/gowalla
```

### 2. Threshold Ablation — ViFoodRec (mean ± std, 3 seed)

| θ | #Interactions | Density | Recall@20 | NDCG@20 | Precision@20 |
|---|----------------|---------|-----------|---------|--------------|
| 3.0 | 82,927 | 0.2052 | 0.0059 ± 0.0004 | 0.0458 ± 0.0036 | 0.0480 ± 0.0032 |
| **3.5** ★ | **66,358** | **0.1642** | **0.0061 ± 0.0008** | **0.0418 ± 0.0058** | **0.0403 ± 0.0055** |
| 4.0 † | 49,670 | 0.1235 | 0.0061 ± 0.0005 | 0.0298 ± 0.0015 | 0.0295 ± 0.0023 |
| 4.5 | 32,109 | 0.0857 | 0.0059 ± 0.0006 | 0.0180 ± 0.0020 | 0.0183 ± 0.0020 |

† Ngưỡng dùng trong nghiên cứu gốc của ViFoodRec.

**→ Chốt θ = 3.5.** Recall@20 gần như tương đương ở θ ∈ {3.0, 3.5, 4.0}, nhưng **NDCG@20** phân biệt rõ rệt: θ=3.5 vượt trội θ=4.0 (0.0418 so với 0.0298), khoảng biến thiên không chồng lấn. Ở θ=4.5, đồ thị quá thưa, mô hình kém hiệu quả.

### 3. Ablation số lớp GCN — K (mean, 3 seed, θ=3.5)

| K | Best epoch | Recall@20 | NDCG@20 | Precision@20 |
|---|-----------|-----------|---------|---------------|
| 1 | 280 | **0.0075** | 0.0508 | **0.0490** |
| 2 | 980 | 0.0070 | 0.0480 | 0.0455 |
| 3 (mặc định) | 720 | 0.0073 | **0.0508** | 0.0475 |
| 4 | 1000 | 0.0070 | **0.0533** | 0.0460 |

Biên độ chênh lệch giữa các K chỉ 0.0005 (Recall) — không có K nào vượt trội rõ rệt; **K=3** được chọn làm mặc định để cân bằng giữa hiệu năng và số epoch hội tụ.

### 4. Kết quả chính thức — So sánh 3 mô hình trên ViFoodRec (θ=3.5, mean ± std, **5 seed**)

| Mô hình | Recall@20 | NDCG@20 | Precision@20 |
|---------|-----------|---------|---------------|
| BPR-MF | 0.0073 ± 0.0003 | 0.0465 ± 0.0021 | 0.0480 ± 0.0019 |
| NGCF | 0.0072 ± 0.0003 | 0.0467 ± 0.0041 | 0.0472 ± 0.0019 |
| **LightGCN** | 0.0067 ± 0.0007 | 0.0439 ± 0.0055 | 0.0441 ± 0.0048 |

**RQ3:** LightGCN thấp hơn BPR-MF/NGCF ~8–9% ở Recall@20, nhưng chênh lệch tuyệt đối (0.0005–0.0006) nhỏ hơn độ lệch chuẩn của chính LightGCN (±0.0007) nên không có ý nghĩa thống kê rõ ràng. Đáng chú ý hơn là **độ ổn định**: LightGCN dao động mạnh hơn hẳn hai baseline (std NDCG@20 gấp ~2.6 lần, std Recall@20 gấp ~2.3 lần BPR-MF/NGCF) — dấu hiệu mô hình graph-based nhạy cảm hơn với khởi tạo ngẫu nhiên khi đồ thị nhỏ và thưa (101 user, 4,001 item).

### 5. Error Analysis

**Long-tail Analysis** (Recall@20, mean ± std, 5 seed — Head: top 20%/800 items, Body: 60%/2400 items, Tail: bottom 20%/801 items):

| Model | Head R@20 | Body R@20 | Tail R@20 | Head/Body Ratio |
|-------|-----------|-----------|-----------|------------------|
| BPR-MF | 0.0136 ± 0.0030 | 0.0070 ± 0.0008 | 0.0025 ± 0.0013 | 1.95× |
| NGCF | 0.0152 ± 0.0020 | 0.0066 ± 0.0006 | 0.0020 ± 0.0006 | 2.30× |
| **LightGCN** | **0.0208 ± 0.0038** | 0.0047 ± 0.0007 | **0.0004 ± 0.0003** | **4.45×** |

→ **Popularity bias rõ rệt nhất ở LightGCN**: cao nhất ở Head nhưng gần như không gợi ý được món Tail — message passing ưu tiên item nhiều kết nối, khiến bias mạnh hơn BPR-MF/NGCF.

**User Group Analysis** (top/bottom 25% theo số tương tác, n=26 mỗi nhóm, mean 5 seed):

| Model | Active R@20 | Less-active R@20 | Chênh lệch tương đối |
|-------|-------------|-------------------|----------------------|
| BPR-MF | 0.0072 ± 0.0013 | 0.0068 ± 0.0008 | 5.9% |
| **NGCF** | **0.0073 ± 0.0007** | 0.0060 ± 0.0005 | **21.7%** |
| LightGCN | 0.0071 ± 0.0011 | 0.0067 ± 0.0012 | 6.0% (khoảng std chồng lấn) |

→ NGCF nhạy với mức độ hoạt động user nhất; LightGCN gần như không phân biệt 2 nhóm. Variance giữa từng user vẫn lớn hơn giữa 2 nhóm (case study: có user active recall=0, user khác recall cao nhất dataset).

**Dish Type Analysis** *(phân tích bổ sung, ngoài phạm vi báo cáo chính thức):*

| Loại | Recall@20 |
|------|-----------|
| Món mặn (3805 items) | 0.0074 |
| Món chay (195 items) | 0.0061 |

---

## ⚙️ Cấu hình thực nghiệm

| Tham số | Giá trị |
|---------|---------|
| Embedding dimension | 64 |
| Số lớp GCN (n_layers, K) | 3 (ablation K ∈ {1,2,3,4}, xem Mục 3) |
| Learning rate | 0.001 |
| L2 regularization (decay) | 1e-4 |
| Batch size | **1024** — dùng chung cho cả BPR-MF / NGCF / LightGCN |
| Epochs (tối đa) | 1000 |
| Train/test split | 80/20 per-user (stratified random split) |
| K-core | 5 |
| Random seeds — ablation threshold & K | 3 seed (2026, 42, 123) |
| Random seeds — kết quả chính thức (so sánh 3 model) | 5 seed |
| GPU | Kaggle Tesla T4 |
| Thời gian train/lần (LightGCN, ViFoodRec) | ~22 phút |

---

## 🔧 Cài đặt & Chạy

### Yêu cầu
Xem `requirements.txt`:
```
torch_geometric
torch
pandas
numpy
matplotlib
scipy
```

### Tiền xử lý
```bash
python scripts/preprocess_vifoodrec.py
# Output: data/processed/experiments/th{3_0,3_5,4_0,4_5}/
```
#### Lưu ý: cần train cả 3 model để có file checkpoint khớp định dạng input cho file longtai_analysis.py

### Train + so sánh cả 3 model (kết quả chính thức)
```bash
python scripts/run_all_baselines.py
# Output: checkpoints/{bprmf,ngcf,lightgcn}_th3_5_seed{seed}.pt
#         outputs/predictions/{model}_th3_5_seed{seed}.npy
```

### Train riêng lẻ từng model

```bash
# LightGCN (PyG)
python scripts/train_lightgcn_pyg.py --seed 2026 --data-dir data/processed/experiments/th3_5

# BPR-MF (custom)
python scripts/train.py --model bprmf --threshold 3.5 --seed 2026

# NGCF (custom)
python scripts/train.py --model ngcf --threshold 3.5 --seed 2026
```
**Lưu ý:** train.py chỉ hỗ trợ BPR-MF và NGCF. LightGCN dùng train_lightgcn_pyg.py vì sử dụng torch_geometric.

### Đánh giá lại Gowalla (không train lại)
```bash
python scripts/evaluate_gowalla_checkpoint.py --ckpt checkpoints/gowalla/lgn-gowalla-3-64.pth.tar --data-dir data/gowalla
```

### Tổng hợp kết quả nhiều seed
```bash
python scripts/summarize_seeds.py
python scripts/evaluate_checkpoint.py
```

### Error Analysis & Long-tail
```bash
python scripts/error_analysis.py        # user group, dish type, case study
python scripts/longtail_analysis.py     # Head/Body/Tail cho cả 3 model
```

### Demo tương tác
Mở trực tiếp `demo/vifoodrec_demo.html` bằng trình duyệt (không cần server) — hiển thị gợi ý món ăn theo user thật (checkpoint LightGCN seed=2026) cùng toàn bộ bảng/biểu đồ ở Mục Kết quả.

---

## Dataset

**ViFoodRec** (Tran et al., PACLIC 2024)
- Repo: [github.com/QuocAn55/A-New-Dataset-and-Empirical-Evaluation-for-Vietnamese-Food-Recommendation-System](https://github.com/QuocAn55/A-New-Dataset-and-Empirical-Evaluation-for-Vietnamese-Food-Recommendation-System)
- `foods.csv`: 4000+ món ăn Việt Nam (tên, loại, calories, cooking_time)
- `ratings.csv`: ~180,000 ratings từ 101 user ( (!) đã fill ~40% giá trị thiếu bằng median per-user — xem `notes/note_report.md` mục 1)

---

## Tài liệu tham khảo

1. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). **LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.** *SIGIR 2020.*
2. Tran, Q. A., Dang, C. T., Dang, T. H. T., et al. (2024). **A New Dataset and Empirical Evaluation for Vietnamese Food Recommendation System.** *PACLIC 2024.*
3. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). **BPR: Bayesian Personalized Ranking from Implicit Feedback.** *UAI 2009 / arXiv 2012.*
4. Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S. (2019). **Neural Graph Collaborative Filtering.** *SIGIR 2019.*
5. Hu, Y., Koren, Y., & Volinsky, C. (2008). **Collaborative filtering for implicit feedback datasets.** *ICDM 2008.*
