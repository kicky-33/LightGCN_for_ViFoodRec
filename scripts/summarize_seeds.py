"""
Tong hop ket qua 3 seed cua threshold 3.5 -> tinh mean +/- std
de dua vao bao cao (Table chinh thuc, phan 4.3.4).

Cach dung:
    1. Train 3 lan voi cung DATA_DIR=data/th3_5 nhung khac SEED:
           SEED = 2026  -> luu checkpoint lightgcn_th3_5_seed2026.pt
           SEED = 42    -> luu checkpoint lightgcn_th3_5_seed42.pt
           SEED = 123   -> luu checkpoint lightgcn_th3_5_seed123.pt
       (chi can doi SEED va ten file luu trong train_lightgcn_pyg.py)

    2. Chay script nay:
           python summarize_seeds.py

Output:
    - In bang mean +/- std ra console (copy thang vao bao cao)
    - Ghi file outputs/results/final_result_th3_5.csv
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn.models import LightGCN

# ===================== CONFIG =====================
DATA_DIR = "checkpoints"
CHECKPOINTS = [
    ("2026", "lightgcn_th3_5_seed2026.pt"),
    ("42", "lightgcn_th3_5_seed42.pt"),
    ("123", "lightgcn_th3_5_seed123.pt"),
]
EMBEDDING_DIM = 64
N_LAYERS = 3
TOPK = 20
OUT_CSV = "outputs/results/final_result_th3_5.csv"
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_interactions(path):
    data = {}
    n_user, n_item = 0, 0
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


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / k
    return recall, precis


def NDCGatK_r(test_data, r, k):
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    idcg = np.sum(test_matrix * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred_data * (1.0 / np.log2(np.arange(2, k + 2))), axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        r.append(np.array(pred).astype("float"))
    return np.array(r)


def evaluate_one(ckpt_path, train_pos, test_pos, num_users, num_items):
    num_nodes = num_users + num_items
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    ckpt_nodes = state_dict["embedding.weight"].shape[0]
    if ckpt_nodes != num_nodes:
        raise ValueError(
            f"Checkpoint {ckpt_path} co num_nodes={ckpt_nodes}, "
            f"data cho num_nodes={num_nodes}. Kiem tra lai DATA_DIR / checkpoint."
        )
    model = LightGCN(
        num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=N_LAYERS
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    train_u, train_i = [], []
    for u, items in train_pos.items():
        for it in items:
            train_u.append(u)
            train_i.append(it + num_users)
    edge_index = torch.tensor(
        np.stack(
            [np.concatenate([train_u, train_i]), np.concatenate([train_i, train_u])]
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        out = model.get_embedding(edge_index)
        user_emb, item_emb = out[:num_users], out[num_users:]
        test_users = list(test_pos.keys())
        groundtruth, topk_items = [], []
        for u in test_users:
            scores = (user_emb[u] @ item_emb.T).clone()
            if u in train_pos:
                scores[train_pos[u]] = -1e10
            topk = torch.topk(scores, TOPK).indices.cpu().numpy()
            topk_items.append(topk)
            groundtruth.append(test_pos[u])

    r = getLabel(groundtruth, topk_items)
    recall, precision = RecallPrecision_ATk(groundtruth, r, TOPK)
    ndcg = NDCGatK_r(groundtruth, r, TOPK)
    n = len(test_users)
    return recall / n, precision / n, ndcg / n


def main():
    data_dir = Path(DATA_DIR)
    train_pos, n_user_tr, n_item_tr = load_interactions(data_dir / "train.txt")
    test_pos, n_user_te, n_item_te = load_interactions(data_dir / "test.txt")
    num_users = max(n_user_tr, n_user_te)
    num_items = max(n_item_tr, n_item_te)
    print(
        f"[Data] threshold=3.5 | users={num_users}, items={num_items}, device={DEVICE}\n"
    )

    rows = []
    recalls, precisions, ndcgs = [], [], []
    for seed_label, ckpt_name in CHECKPOINTS:
        ckpt_path = Path(ckpt_name)
        if not ckpt_path.exists():
            print(
                f"[!] Khong tim thay {ckpt_name} — bo qua. "
                f"(Train xong roi chay lai script nay)"
            )
            continue
        recall, precision, ndcg = evaluate_one(
            ckpt_path, train_pos, test_pos, num_users, num_items
        )
        print(
            f"  seed={seed_label:<6} | Recall@{TOPK}={recall:.4f}  "
            f"Precision@{TOPK}={precision:.4f}  NDCG@{TOPK}={ndcg:.4f}"
        )
        recalls.append(recall)
        precisions.append(precision)
        ndcgs.append(ndcg)
        rows.append(
            {
                "seed": seed_label,
                f"recall@{TOPK}": round(recall, 4),
                f"precision@{TOPK}": round(precision, 4),
                f"ndcg@{TOPK}": round(ndcg, 4),
            }
        )

    if not recalls:
        print("\n[!] Chua co checkpoint nao. Train xong roi chay lai script nay.")
        return

    r_mean, r_std = np.mean(recalls), np.std(recalls)
    p_mean, p_std = np.mean(precisions), np.std(precisions)
    n_mean, n_std = np.mean(ndcgs), np.std(ndcgs)

    print(f"\n{'='*60}")
    print(f"  KET QUA CHINH THUC — LightGCN tren ViFoodRec (threshold=3.5)")
    print(f"  So seed: {len(recalls)}/3")
    print(f"{'='*60}")
    print(f"  Recall@{TOPK}    = {r_mean:.4f} ± {r_std:.4f}")
    print(f"  Precision@{TOPK} = {p_mean:.4f} ± {p_std:.4f}")
    print(f"  NDCG@{TOPK}      = {n_mean:.4f} ± {n_std:.4f}")
    print(f"{'='*60}")
    print(f"\n  -> Dua vao bao cao (Table 4.x):")
    print(
        f"     LightGCN (ViFoodRec, th=3.5)  "
        f"{r_mean:.4f}±{r_std:.4f}  "
        f"{p_mean:.4f}±{p_std:.4f}  "
        f"{n_mean:.4f}±{n_std:.4f}"
    )

    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "seed": "mean",
                f"recall@{TOPK}": round(r_mean, 4),
                f"precision@{TOPK}": round(p_mean, 4),
                f"ndcg@{TOPK}": round(n_mean, 4),
            }
        )
        writer.writerow(
            {
                "seed": "std",
                f"recall@{TOPK}": round(r_std, 4),
                f"precision@{TOPK}": round(p_std, 4),
                f"ndcg@{TOPK}": round(n_std, 4),
            }
        )
    print(f"\n  Da ghi: {out_path}")


if __name__ == "__main__":
    main()
