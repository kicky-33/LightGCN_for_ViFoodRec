"""
longtail_analysis.py
=====================
Tính Head / Body / Tail Recall@20 cho 3 model x N seed
từ các file predictions đã lưu bởi run_all_baselines.py.

Cấu trúc file predictions (khớp quy ước path đã thống nhất toàn repo):
    outputs/predictions/{model}_th{threshold}_seed{seed}.npy

Cách dùng:
    python longtail_analysis.py
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CẤU HÌNH — chỉnh nếu đường dẫn khác
# ─────────────────────────────────────────────
THRESHOLD = "3.5"
_TH_DIR = f"th{THRESHOLD.replace('.', '_')}"  # '3.5' -> 'th3_5'
TRAIN_PATH = f"data/processed/experiments/{_TH_DIR}/train.txt"
TEST_PATH = f"data/processed/experiments/{_TH_DIR}/test.txt"

# Đổi theo quy ước path đã chốt: outputs/predictions/, không còn checkpoints/predictions/
PRED_DIR = "outputs/predictions"
OUT_DIR = "outputs/error_analysis"  # khớp cấu trúc thư mục trong README
OUT_CSV = os.path.join(OUT_DIR, "longtail_per_seed.csv")

MODELS = ["bprmf", "ngcf", "lightgcn"]
SEEDS = [2026, 42, 123]  # giữ nguyên — KHÔNG đổi số seed ở đây
K = 20

HEAD_RATIO = 0.20  # top 20% phổ biến nhất
TAIL_RATIO = 0.20  # bottom 20% ít phổ biến nhất


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_interactions(path):
    """Đọc file train/test.txt → dict {user_id: [item_ids]}"""
    user_items = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 1:
                continue
            u = int(parts[0])
            items = [int(x) for x in parts[1:] if x]
            if items:
                user_items[u] = items
    return user_items


# ─────────────────────────────────────────────
# Xây item groups từ train
# ─────────────────────────────────────────────
def build_item_groups(train_user_items):
    item_counts = Counter()
    for items in train_user_items.values():
        for item in items:
            item_counts[item] += 1

    all_items_sorted = sorted(
        item_counts.keys(), key=lambda x: item_counts[x], reverse=True
    )
    n = len(all_items_sorted)
    head_cut = int(HEAD_RATIO * n)
    tail_cut = int((1 - TAIL_RATIO) * n)

    head_items = set(all_items_sorted[:head_cut])
    body_items = set(all_items_sorted[head_cut:tail_cut])
    tail_items = set(all_items_sorted[tail_cut:])

    print(f"[LongTail] Tổng {n} items trong train")
    print(
        f"           Head={len(head_items)} | "
        f"Body={len(body_items)} | Tail={len(tail_items)}"
    )
    return head_items, body_items, tail_items


# ─────────────────────────────────────────────
# Tính recall theo nhóm
# ─────────────────────────────────────────────
def compute_longtail_recall(
    predictions, test_user_items, head_items, body_items, tail_items
):
    """
    predictions: dict {user_id: np.array top-K items}
    Trả về (head_mean, head_std), (body_mean, body_std), (tail_mean, tail_std)
    """
    head_r, body_r, tail_r = [], [], []

    for u, pred in predictions.items():
        gt = set(test_user_items.get(u, []))
        if not gt:
            continue
        pred_set = set(pred[:K])

        gt_head = gt & head_items
        gt_body = gt & body_items
        gt_tail = gt & tail_items

        if gt_head:
            head_r.append(len(pred_set & gt_head) / len(gt_head))
        if gt_body:
            body_r.append(len(pred_set & gt_body) / len(gt_body))
        if gt_tail:
            tail_r.append(len(pred_set & gt_tail) / len(gt_tail))

    def stats(lst):
        if not lst:
            return 0.0, 0.0
        return float(np.mean(lst)), float(np.std(lst))

    return stats(head_r), stats(body_r), stats(tail_r)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LONG-TAIL ANALYSIS — Head / Body / Tail Recall@20")
    print("=" * 60)

    # Load data
    train_user_items = load_interactions(TRAIN_PATH)
    test_user_items = load_interactions(TEST_PATH)
    print(
        f"[Data] {len(train_user_items)} train users, "
        f"{len(test_user_items)} test users"
    )

    # Build groups — dùng chung cho tất cả model/seed
    head_items, body_items, tail_items = build_item_groups(train_user_items)

    # Thu thập kết quả
    per_seed_rows = []  # mỗi model/seed một dòng
    missing_files = []

    for model_name in MODELS:
        print(f"\n── {model_name.upper()} ──")
        h_list, b_list, t_list = [], [], []

        for seed in SEEDS:
            pred_path = os.path.join(
                PRED_DIR, f"{model_name}_{_TH_DIR}_seed{seed}.npy"
            )

            if not os.path.exists(pred_path):
                print(f"  [MISSING] {pred_path}")
                missing_files.append(pred_path)
                continue

            # Load predictions
            predictions = np.load(pred_path, allow_pickle=True).item()

            # Tính recall
            (hm, hs), (bm, bs), (tm, ts) = compute_longtail_recall(
                predictions, test_user_items, head_items, body_items, tail_items
            )

            print(
                f"  seed={seed} | Head={hm:.4f}±{hs:.4f} | "
                f"Body={bm:.4f}±{bs:.4f} | Tail={tm:.4f}±{ts:.4f}"
            )

            h_list.append(hm)
            b_list.append(bm)
            t_list.append(tm)

            per_seed_rows.append(
                {
                    "model": model_name,
                    "threshold": THRESHOLD,
                    "seed": seed,
                    "head": hm,
                    "body": bm,
                    "tail": tm,
                }
            )

        if h_list:
            print(
                f"  MEAN | Head={np.mean(h_list):.4f}±{np.std(h_list):.4f} | "
                f"Body={np.mean(b_list):.4f}±{np.std(b_list):.4f} | "
                f"Tail={np.mean(t_list):.4f}±{np.std(t_list):.4f}"
            )

    # ── Bảng tổng hợp ─────────────────────────────────────
    if per_seed_rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        df = pd.DataFrame(per_seed_rows)
        df.to_csv(OUT_CSV, index=False)

        print(f"\n{'='*60}")
        print(f"BẢNG TỔNG HỢP (mean ± std, n={len(SEEDS)} seed)")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Head R@20':>12} {'Body R@20':>12} {'Tail R@20':>12}")
        print("-" * 50)
        for model_name in MODELS:
            sub = df[df["model"] == model_name]
            if sub.empty:
                continue
            print(
                f"{model_name.upper():<12} "
                f"{sub.head.mean():.4f}±{sub.head.std():.4f}  "
                f"{sub.body.mean():.4f}±{sub.body.std():.4f}  "
                f"{sub.tail.mean():.4f}±{sub.tail.std():.4f}"
            )

        print(f"\n[INFO] Đã lưu: {OUT_CSV}")

    if missing_files:
        print(f"\n[WARN] Thiếu {len(missing_files)} file predictions:")
        for f in missing_files:
            print(f"  {f}")
        print("  → Chạy run_all_baselines.py với model/seed tương ứng trước.")


if __name__ == "__main__":
    main()
