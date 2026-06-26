"""
Danh gia 1 checkpoint LightGCN da train (.pt) tren tap test, KHONG train lai.

LY DO CAN SCRIPT NAY: checkpoint .pt chi luu trong so model (state_dict),
KHONG luu Recall@20/NDCG@20/Precision@20 da in ra console luc train.
-> Neu khong con log console, dung script nay de tinh lai metric tu checkpoint.

Cach dung:
    Sua CKPT_PATH va DATA_DIR ben duoi roi chay:
        python evaluate_checkpoint.py
    Hoac chay nhieu threshold lien tiep bang vong lap shell (xem huong dan o cuoi file).

Ket qua duoc APPEND vao outputs/results/threshold_ablation_results.csv
(khong ghi de, moi lan chay them 1 dong) de tong hop dan ca 4 threshold.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn.models import LightGCN

# ===================== CONFIG =====================
CKPT_PATH = "checkpoints/lightgcn_th4_5.pt"  # file checkpoint can danh gia
DATA_DIR = "data/processed/experiments/th4_5"  # thu muc chua train.txt/test.txt TUONG UNG checkpoint nay
THRESHOLD_LABEL = "4.5"  # ghi nhan threshold nao, de dien vao bang ket qua
EMBEDDING_DIM = 64
N_LAYERS = 3
TOPK = 20
RESULTS_CSV = "outputs/results/threshold_ablation_results.csv"
# ====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_interactions(path):
    data = {}
    n_user, n_item = 0, 0
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


# ---------- Metric - copy nguyen tu gusye1234/LightGCN-PyTorch/code/utils.py (giong train script) ----------
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


def main():
    data_dir = Path(DATA_DIR)
    train_pos, n_user_tr, n_item_tr = load_interactions(data_dir / "train.txt")
    test_pos, n_user_te, n_item_te = load_interactions(data_dir / "test.txt")
    num_users = max(n_user_tr, n_user_te)
    num_items = max(n_item_tr, n_item_te)
    num_nodes = num_users + num_items

    # ---------- Doi chieu checkpoint voi data truoc khi danh gia (tranh nham lan threshold) ----------
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    ckpt_num_nodes, ckpt_emb_dim = state_dict["embedding.weight"].shape
    if ckpt_num_nodes != num_nodes:
        raise ValueError(
            f"[!] Checkpoint co num_nodes={ckpt_num_nodes} nhung data o {DATA_DIR} "
            f"cho num_nodes={num_nodes} (users={num_users}, items={num_items}). "
            f"Kiem tra lai ban da tro dung CKPT_PATH/DATA_DIR cua CUNG 1 threshold chua."
        )
    print(
        f"[OK] Checkpoint khop voi data: num_nodes={num_nodes} (users={num_users}, items={num_items})"
    )

    model = LightGCN(
        num_nodes=num_nodes, embedding_dim=ckpt_emb_dim, num_layers=N_LAYERS
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
        recall, precision, ndcg = recall / n, precision / n, ndcg / n

    print(
        f"[RESULT] threshold={THRESHOLD_LABEL} | Recall@{TOPK}={recall:.4f} "
        f"Precision@{TOPK}={precision:.4f} NDCG@{TOPK}={ndcg:.4f}"
    )

    out_path = Path(RESULTS_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "threshold",
                    "n_users",
                    "n_items",
                    f"recall@{TOPK}",
                    f"precision@{TOPK}",
                    f"ndcg@{TOPK}",
                    "checkpoint",
                ]
            )
        writer.writerow(
            [
                THRESHOLD_LABEL,
                num_users,
                num_items,
                round(recall, 4),
                round(precision, 4),
                round(ndcg, 4),
                CKPT_PATH,
            ]
        )
    print(f"Da ghi vao {out_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Chay lien tiep ca 4 threshold (PowerShell, sua duong dan cho khop may ban):
#
#   $configs = @(
#     @{th="3.0"; ckpt="lightgcn_th3_0.pt"; data="data/th3_0"},
#     @{th="3.5"; ckpt="lightgcn_th3_5.pt"; data="data/th3_5"},
#     @{th="4.0"; ckpt="lightgcn_th4_0.pt"; data="data/th4_0"},
#     @{th="4.5"; ckpt="lightgcn_th4_5.pt"; data="data/th4_5"}
#   )
#   foreach ($c in $configs) {
#     (Get-Content evaluate_checkpoint.py) `
#       -replace 'CKPT_PATH = ".*"', "CKPT_PATH = `"$($c.ckpt)`"" `
#       -replace 'DATA_DIR = ".*"', "DATA_DIR = `"$($c.data)`"" `
#       -replace 'THRESHOLD_LABEL = ".*"', "THRESHOLD_LABEL = `"$($c.th)`"" `
#       | Set-Content evaluate_checkpoint_tmp.py
#     python evaluate_checkpoint_tmp.py
#   }
# ---------------------------------------------------------------------------
