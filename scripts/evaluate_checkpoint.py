"""
Danh gia 12 checkpoint (4 threshold x 3 seed) trong 1 lan chay.
Output: outputs/results/threshold_ablation_results.csv  (tung dong per checkpoint)
        outputs/results/threshold_ablation_summary.csv  (mean+-std per threshold)
"""

import csv
import numpy as np
import torch
from pathlib import Path
from torch_geometric.nn.models import LightGCN

# ===================== CONFIG =====================
EMBEDDING_DIM = 64
N_LAYERS = 3
TOPK = 20
RESULTS_CSV = "outputs/results/threshold_ablation_results.csv"
SUMMARY_CSV = "outputs/results/threshold_ablation_summary.csv"

CONFIGS = [
    {
        "threshold": "3.0",
        "seed": 2026,
        "ckpt": "checkpoints/lightgcn_th3_0_seed2026.pt",
        "data": "data/processed/experiments/th3_0",
    },
    {
        "threshold": "3.0",
        "seed": 42,
        "ckpt": "checkpoints/lightgcn_th3_0_seed42.pt",
        "data": "data/processed/experiments/th3_0",
    },
    {
        "threshold": "3.0",
        "seed": 123,
        "ckpt": "checkpoints/lightgcn_th3_0_seed123.pt",
        "data": "data/processed/experiments/th3_0",
    },
    {
        "threshold": "3.5",
        "seed": 2026,
        "ckpt": "checkpoints/lightgcn_th3_5_seed2026.pt",
        "data": "data/processed/experiments/th3_5",
    },
    {
        "threshold": "3.5",
        "seed": 42,
        "ckpt": "checkpoints/lightgcn_th3_5_seed42.pt",
        "data": "data/processed/experiments/th3_5",
    },
    {
        "threshold": "3.5",
        "seed": 123,
        "ckpt": "checkpoints/lightgcn_th3_5_seed123.pt",
        "data": "data/processed/experiments/th3_5",
    },
    {
        "threshold": "4.0",
        "seed": 2026,
        "ckpt": "checkpoints/lightgcn_th4_0_seed2026.pt",
        "data": "data/processed/experiments/th4_0",
    },
    {
        "threshold": "4.0",
        "seed": 42,
        "ckpt": "checkpoints/lightgcn_th4_0_seed42.pt",
        "data": "data/processed/experiments/th4_0",
    },
    {
        "threshold": "4.0",
        "seed": 123,
        "ckpt": "checkpoints/lightgcn_th4_0_seed123.pt",
        "data": "data/processed/experiments/th4_0",
    },
    {
        "threshold": "4.5",
        "seed": 2026,
        "ckpt": "checkpoints/lightgcn_th4_5_seed2026.pt",
        "data": "data/processed/experiments/th4_5",
    },
    {
        "threshold": "4.5",
        "seed": 42,
        "ckpt": "checkpoints/lightgcn_th4_5_seed42.pt",
        "data": "data/processed/experiments/th4_5",
    },
    {
        "threshold": "4.5",
        "seed": 123,
        "ckpt": "checkpoints/lightgcn_th4_5_seed123.pt",
        "data": "data/processed/experiments/th4_5",
    },
]
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_interactions(path):
    data, n_user, n_item = {}, 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(" ")
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
    dcg = np.sum(pred_data * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.0] = 1.0
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.0
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        pred = list(map(lambda x: x in test_data[i], pred_data[i]))
        r.append(np.array(pred).astype("float"))
    return np.array(r)


def evaluate_one(cfg, data_cache):
    ckpt_path = Path(cfg["ckpt"])
    if not ckpt_path.exists():
        print(f"  [SKIP] Chua co: {ckpt_path}")
        return None

    data_dir = cfg["data"]
    if data_dir not in data_cache:
        train_pos, nu_tr, ni_tr = load_interactions(Path(data_dir) / "train.txt")
        test_pos, nu_te, ni_te = load_interactions(Path(data_dir) / "test.txt")
        num_users = max(nu_tr, nu_te)
        num_items = max(ni_tr, ni_te)
        data_cache[data_dir] = (train_pos, test_pos, num_users, num_items)

    train_pos, test_pos, num_users, num_items = data_cache[data_dir]
    num_nodes = num_users + num_items

    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    ckpt_nodes = state_dict["embedding.weight"].shape[0]
    if ckpt_nodes != num_nodes:
        raise ValueError(
            f"[!] {ckpt_path.name}: num_nodes={ckpt_nodes} "
            f"!= data num_nodes={num_nodes}. Kiem tra lai cap ckpt/data."
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
        groundtruth, topk_items = [], []
        for u in test_pos:
            scores = (user_emb[u] @ item_emb.T).clone()
            if u in train_pos:
                scores[train_pos[u]] = -1e10
            topk = torch.topk(scores, TOPK).indices.cpu().numpy()
            topk_items.append(topk)
            groundtruth.append(test_pos[u])

    r = getLabel(groundtruth, topk_items)
    recall, precision = RecallPrecision_ATk(groundtruth, r, TOPK)
    ndcg = NDCGatK_r(groundtruth, r, TOPK)
    n = len(test_pos)
    return round(recall / n, 6), round(precision / n, 6), round(ndcg / n, 6)


def main():
    Path(RESULTS_CSV).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    data_cache = {}

    print(f"Device: {DEVICE}")
    print(
        f"{'Threshold':>10} {'Seed':>6} {'Recall':>8} {'Prec':>8} {'NDCG':>8}  Checkpoint"
    )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "seed",
                "n_users",
                "n_items",
                f"recall@{TOPK}",
                f"precision@{TOPK}",
                f"ndcg@{TOPK}",
                "checkpoint",
            ]
        )

        for cfg in CONFIGS:
            result = evaluate_one(cfg, data_cache)
            if result is None:
                continue
            recall, precision, ndcg = result
            th, seed = cfg["threshold"], cfg["seed"]
            _, _, num_users, num_items = data_cache[cfg["data"]]

            print(
                f"{th:>10} {seed:>6} {recall:>8.4f} {precision:>8.4f} {ndcg:>8.4f}  {Path(cfg['ckpt']).name}"
            )
            writer.writerow(
                [th, seed, num_users, num_items, recall, precision, ndcg, cfg["ckpt"]]
            )
            rows.append(
                {
                    "threshold": th,
                    "recall": recall,
                    "precision": precision,
                    "ndcg": ndcg,
                }
            )

    # mean+-std per threshold
    from collections import defaultdict

    by_th = defaultdict(list)
    for r in rows:
        by_th[r["threshold"]].append(r)

    print(f"\n{'':=<70}")
    print(
        f"{'Threshold':>10} {'n_seeds':>8} {'Recall (mean+-std)':>22} {'NDCG (mean+-std)':>22}"
    )
    print(f"{'':=<70}")

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "n_seeds",
                "recall_mean",
                "recall_std",
                "ndcg_mean",
                "ndcg_std",
                "precision_mean",
                "precision_std",
            ]
        )

        for th in ["3.0", "3.5", "4.0", "4.5"]:
            if th not in by_th:
                continue
            vals = by_th[th]
            r_arr = np.array([v["recall"] for v in vals])
            p_arr = np.array([v["precision"] for v in vals])
            n_arr = np.array([v["ndcg"] for v in vals])
            rm, rs = r_arr.mean(), r_arr.std()
            pm, ps = p_arr.mean(), p_arr.std()
            nm, ns = n_arr.mean(), n_arr.std()
            print(
                f"{th:>10} {len(vals):>8} "
                f"{rm:.4f}+-{rs:.4f}        "
                f"{nm:.4f}+-{ns:.4f}"
            )
            writer.writerow(
                [
                    th,
                    len(vals),
                    round(rm, 6),
                    round(rs, 6),
                    round(nm, 6),
                    round(ns, 6),
                    round(pm, 6),
                    round(ps, 6),
                ]
            )

    print(f"\nDa ghi:\n  {RESULTS_CSV}\n  {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
