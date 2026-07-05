"""
ERROR ANALYSIS — LightGCN tren ViFoodRec (threshold=3.5)

Phan tich 4 khia canh:
    1. Long-tail analysis     — hieu suat theo do pho bien cua mon an
    2. User group analysis    — active vs less-active user
    3. Dish type analysis     — Mon man vs Mon chay
    4. Case study             — Top-10 goi y cho 2-3 user cu the

Input can co:
    - DATA_DIR/train.txt, test.txt   (cua threshold=3.5)
    - CKPT_PATH                      (checkpoint tot nhat, de nghi seed=2026)
    - FOODS_CSV                      (foods.csv tu repo ViFoodRec)
    - FOOD_ID_MAP                    (foodid_map.csv tu preprocess_vifoodrec.py)

Output (trong OUTPUT_DIR):
    - error_analysis_longtail.png
    - error_analysis_usergroup.png
    - error_analysis_dishtype.png
    - case_study.csv                 (dua thang vao bao cao)
    - error_analysis_summary.txt     (so lieu tong hop, copy vao phan 4.5)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.nn.models import LightGCN

# ===================== CONFIG =====================
seed = 2026
DATA_DIR = "data/processed/experiments/th3_5"
CKPT_PATH = f"checkpoints/lightgcn_th3_5_seed{seed}.pt"
FOODS_CSV = "data/raw/foods.csv"
FOOD_ID_MAP = "data/processed/experiments/th3_5/foodid_map.csv"
OUTPUT_DIR = Path("outputs/error_analysis") / f"seed{seed}"
TOPK = 20
EMBEDDING_DIM = 64
N_LAYERS = 3
CASE_STUDY_USERS = 3  # so user chon de phan tich case study
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────── helpers ─────────────────────────
def load_interactions(path):
    data, n_user, n_item = {}, 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            if not items:
                continue
            data[u] = items
            n_user = max(n_user, u)
            n_item = max(n_item, max(items))
    return data, n_user + 1, n_item + 1


def get_model_and_embeddings(ckpt, train_pos, num_users, num_items):
    num_nodes = num_users + num_items
    sd = torch.load(ckpt, map_location=DEVICE)
    model = LightGCN(
        num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=N_LAYERS
    ).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    train_u, train_i = [], []
    for u, items in train_pos.items():
        for it in items:
            train_u.append(u)
            train_i.append(it + num_users)
    ei = torch.tensor(
        np.stack(
            [np.concatenate([train_u, train_i]), np.concatenate([train_i, train_u])]
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        out = model.get_embedding(ei)
    user_emb = out[:num_users]
    item_emb = out[num_users:]
    return user_emb, item_emb, train_pos


def get_topk_per_user(user_emb, item_emb, train_pos, test_pos, k):
    """Tra ve dict {user: topk_item_indices} (da loai train items)."""
    results = {}
    for u in test_pos:
        scores = (user_emb[u] @ item_emb.T).clone()
        if u in train_pos:
            scores[train_pos[u]] = -1e10
        results[u] = torch.topk(scores, k).indices.cpu().numpy().tolist()
    return results


def recall_ndcg_at_k(topk_list, groundtruth, k):
    hits = [1 if it in groundtruth else 0 for it in topk_list[:k]]
    recall = sum(hits) / len(groundtruth) if groundtruth else 0
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(groundtruth), k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return recall, ndcg


def save_fig(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> Da luu: {p}")


# ───────────────────────── main ─────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    # ---------- Load data ----------
    train_pos, nu_tr, ni_tr = load_interactions(Path(DATA_DIR) / "train.txt")
    test_pos, nu_te, ni_te = load_interactions(Path(DATA_DIR) / "test.txt")
    num_users = max(nu_tr, nu_te)
    num_items = max(ni_tr, ni_te)
    print(f"[Data] users={num_users}, items={num_items}")

    user_emb, item_emb, _ = get_model_and_embeddings(
        CKPT_PATH, train_pos, num_users, num_items
    )
    topk_preds = get_topk_per_user(user_emb, item_emb, train_pos, test_pos, TOPK)

    # ---------- Load foods & mapping ----------
    foods = pd.read_csv(FOODS_CSV)[
        ["food_id", "dish_name", "dish_type", "calories", "cooking_time"]
    ]
    id_map = pd.read_csv(FOOD_ID_MAP)
    # food_id_map: foodid_goc -> foodidx
    goc_to_idx = dict(zip(id_map["foodid_goc"], id_map["foodidx"]))
    idx_to_goc = {v: k for k, v in goc_to_idx.items()}
    foods["foodidx"] = foods["food_id"].map(goc_to_idx)
    foods = foods.dropna(subset=["foodidx"])
    foods["foodidx"] = foods["foodidx"].astype(int)
    food_info = foods.set_index("foodidx")

    # ---------- Item popularity (so lan xuat hien trong train) ----------
    item_pop = {}
    for items in train_pos.values():
        for it in items:
            item_pop[it] = item_pop.get(it, 0) + 1
    all_items_sorted = sorted(item_pop, key=lambda x: -item_pop.get(x, 0))
    n_items_pop = len(all_items_sorted)
    head_set = set(all_items_sorted[: int(n_items_pop * 0.2)])
    tail_set = set(all_items_sorted[int(n_items_pop * 0.8) :])
    body_set = set(all_items_sorted) - head_set - tail_set

    # ──────────────────────────────────────────────
    # 1. LONG-TAIL ANALYSIS
    # ──────────────────────────────────────────────
    print("\n[1] Long-tail analysis...")
    group_metrics = {"Head (top 20%)": [], "Body (60%)": [], "Tail (bottom 20%)": []}
    group_hit_counts = {"Head (top 20%)": 0, "Body (60%)": 0, "Tail (bottom 20%)": 0}
    group_test_counts = {"Head (top 20%)": 0, "Body (60%)": 0, "Tail (bottom 20%)": 0}

    for u, gt in test_pos.items():
        pred = topk_preds[u]
        for group_name, group_set in [
            ("Head (top 20%)", head_set),
            ("Body (60%)", body_set),
            ("Tail (bottom 20%)", tail_set),
        ]:
            gt_g = [it for it in gt if it in group_set]
            pred_g = [it for it in pred if it in group_set]
            if not gt_g:
                continue
            hits = len(set(pred_g) & set(gt_g))
            group_hit_counts[group_name] += hits
            group_test_counts[group_name] += len(gt_g)

    longtail_recall = {}
    for g in group_metrics:
        r = (
            group_hit_counts[g] / group_test_counts[g]
            if group_test_counts[g] > 0
            else 0
        )
        longtail_recall[g] = round(r, 4)
        print(
            f"  {g}: Recall@{TOPK}={r:.4f} "
            f"(hits={group_hit_counts[g]}, test_items={group_test_counts[g]})"
        )

    summary_lines += [
        "\n=== 1. LONG-TAIL ANALYSIS ===",
        f"Phan loai: Head=top 20% pho bien ({len(head_set)} items), "
        f"Body=60% ({len(body_set)} items), Tail=bottom 20% ({len(tail_set)} items)",
    ] + [f"  {g}: Recall@{TOPK}={v}" for g, v in longtail_recall.items()]

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = list(longtail_recall.keys())
    vals = list(longtail_recall.values())
    bars = ax.bar(
        groups,
        vals,
        color=["#378ADD", "#1D9E75", "#D85A30"],
        width=0.5,
        edgecolor="white",
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.0002,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel(f"Recall@{TOPK}", fontsize=11)
    ax.set_title("Long-tail Analysis: Recall theo độ phổ biến món ăn", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.patch.set_facecolor("white")
    save_fig(fig, "error_analysis_longtail.png")

    # ──────────────────────────────────────────────
    # 2. USER GROUP ANALYSIS
    # ──────────────────────────────────────────────
    print("\n[2] User group analysis...")
    user_activity = {u: len(items) for u, items in train_pos.items()}

    q75 = np.percentile(list(user_activity.values()), 75)
    q25 = np.percentile(list(user_activity.values()), 25)
    active_users = {u for u, n in user_activity.items() if n >= q75}
    less_users = {u for u, n in user_activity.items() if n <= q25}

    def group_recall_ndcg(user_set):
        rs, ns = [], []
        for u in user_set:
            if u not in test_pos:
                continue
            r, n = recall_ndcg_at_k(topk_preds[u], test_pos[u], TOPK)
            rs.append(r)
            ns.append(n)
        return np.mean(rs) if rs else 0, np.mean(ns) if ns else 0

    r_act, n_act = group_recall_ndcg(active_users)
    r_less, n_less = group_recall_ndcg(less_users)
    print(
        f"  Active (>= P75={q75:.0f} interactions): "
        f"Recall={r_act:.4f}, NDCG={n_act:.4f} ({len(active_users)} users)"
    )
    print(
        f"  Less-active (<= P25={q25:.0f}): "
        f"Recall={r_less:.4f}, NDCG={n_less:.4f} ({len(less_users)} users)"
    )

    summary_lines += [
        "\n=== 2. USER GROUP ANALYSIS ===",
        f"P25={q25:.0f}, P75={q75:.0f} interactions per user (train)",
        f"Active users (>= P75={q75:.0f}): n={len(active_users)}, "
        f"Recall@{TOPK}={r_act:.4f}, NDCG@{TOPK}={n_act:.4f}",
        f"Less-active users (<= P25={q25:.0f}): n={len(less_users)}, "
        f"Recall@{TOPK}={r_less:.4f}, NDCG@{TOPK}={n_less:.4f}",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric_name, vals in zip(
        axes, [f"Recall@{TOPK}", f"NDCG@{TOPK}"], [(r_act, r_less), (n_act, n_less)]
    ):
        labels = [
            f"Active\n(n={len(active_users)})",
            f"Less-active\n(n={len(less_users)})",
        ]
        bars = ax.bar(
            labels, vals, color=["#378ADD", "#D85A30"], width=0.45, edgecolor="white"
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.0001,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_title(metric_name, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.3)
        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.suptitle("User Group Analysis: Active vs Less-active", fontsize=11)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    save_fig(fig, "error_analysis_usergroup.png")

    # ──────────────────────────────────────────────
    # 3. DISH TYPE ANALYSIS (Mon man vs Mon chay)
    # ──────────────────────────────────────────────
    print("\n[3] Dish type analysis...")
    man_items = set(food_info[food_info["dish_type"] == "Món mặn"].index)
    chay_items = set(food_info[food_info["dish_type"] == "Món chay"].index)

    def dish_recall(item_set, label):
        hits, total = 0, 0
        for u, gt in test_pos.items():
            gt_g = [it for it in gt if it in item_set]
            if not gt_g:
                continue
            pred_g = [it for it in topk_preds[u] if it in item_set]
            hits += len(set(pred_g) & set(gt_g))
            total += len(gt_g)
        r = hits / total if total > 0 else 0
        print(f"  {label}: Recall@{TOPK}={r:.4f} (hits={hits}, test_items={total})")
        return r

    r_man = dish_recall(man_items, "Món mặn")
    r_chay = dish_recall(chay_items, "Món chay")

    summary_lines += [
        "\n=== 3. DISH TYPE ANALYSIS ===",
        f"Mon man ({len(man_items)} items in map): Recall@{TOPK}={r_man:.4f}",
        f"Mon chay ({len(chay_items)} items in map): Recall@{TOPK}={r_chay:.4f}",
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Món mặn", "Món chay"],
        [r_man, r_chay],
        color=["#378ADD", "#1D9E75"],
        width=0.4,
        edgecolor="white",
    )
    for b, v in zip(bars, [r_man, r_chay]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.0002,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax.set_ylabel(f"Recall@{TOPK}", fontsize=11)
    ax.set_title("Dish Type Analysis: Món mặn vs Món chay", fontsize=11)
    ax.set_ylim(0, max(r_man, r_chay) * 1.3)
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.patch.set_facecolor("white")
    save_fig(fig, "error_analysis_dishtype.png")

    # ──────────────────────────────────────────────
    # 4. CASE STUDY
    # ──────────────────────────────────────────────
    print(f"\n[4] Case study ({CASE_STUDY_USERS} users)...")
    # Chon 1 active user co Recall tot, 1 active user Recall kem, 1 less-active
    user_recalls = {}
    for u in test_pos:
        r, _ = recall_ndcg_at_k(topk_preds[u], test_pos[u], TOPK)
        user_recalls[u] = r

    active_sorted = sorted(active_users & set(test_pos), key=lambda u: -user_recalls[u])
    less_sorted = sorted(less_users & set(test_pos), key=lambda u: -user_recalls[u])

    chosen = []
    if active_sorted:
        chosen.append(("Active — Recall cao", active_sorted[0]))
    if len(active_sorted) > 1:
        chosen.append(("Active — Recall thap", active_sorted[-1]))
    if less_sorted:
        chosen.append(("Less-active", less_sorted[len(less_sorted) // 2]))
    chosen = chosen[:CASE_STUDY_USERS]

    case_rows = []
    for role, u in chosen:
        u_act = user_activity.get(u, 0)
        u_recall = user_recalls[u]
        pred_items = topk_preds[u][:10]
        gt_items = test_pos[u]

        for rank, it in enumerate(pred_items, 1):
            orig_id = idx_to_goc.get(it)
            if orig_id is not None and orig_id in food_info.index:
                info = food_info.loc[orig_id]
                name = info["dish_name"]
                dtype = info["dish_type"]
                cal = info["calories"]
            else:
                name, dtype, cal = f"item_idx={it}", "N/A", "N/A"
            hit = "✓" if it in gt_items else ""
            case_rows.append(
                {
                    "user_role": role,
                    "user_idx": u,
                    "n_train_interactions": u_act,
                    f"recall@{TOPK}": round(u_recall, 4),
                    "rank": rank,
                    "item_idx": it,
                    "dish_name": name,
                    "dish_type": dtype,
                    "calories": cal,
                    "hit": hit,
                }
            )

    case_df = pd.DataFrame(case_rows)
    case_path = OUTPUT_DIR / "case_study.csv"
    case_df.to_csv(case_path, index=False, encoding="utf-8-sig")
    print(f"  -> Da luu: {case_path}")

    # In preview ra console
    for role, u in chosen:
        sub = case_df[case_df["user_idx"] == u]
        print(
            f"\n  [{role}] user={u}, "
            f"n_train={user_activity.get(u,0)}, "
            f"Recall@{TOPK}={user_recalls[u]:.4f}"
        )
        for _, row in sub.iterrows():
            print(
                f"    #{row['rank']:2d} {row['hit']:1s} {row['dish_name'][:35]:<35} "
                f"({row['dish_type']}, {row['calories']} kcal)"
            )

    summary_lines += [
        "\n=== 4. CASE STUDY ===",
        f"Chi tiet xem file: {case_path}",
    ] + [
        f"  {role}: user={u}, n_train={user_activity.get(u,0)}, "
        f"Recall@{TOPK}={user_recalls[u]:.4f}"
        for role, u in chosen
    ]

    # ---------- Ghi summary ----------
    summary_path = OUTPUT_DIR / "error_analysis_summary.txt"
    summary_path.write_text(
        f"=== ERROR ANALYSIS SUMMARY — LightGCN ViFoodRec (th=3.5, seed={seed}) ===\n"
        + "\n".join(summary_lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n[Done] Tat ca ket qua o: {OUTPUT_DIR}/")
    print("       Copy error_analysis_summary.txt vao phan 4.5 bao cao.")


if __name__ == "__main__":
    main()
