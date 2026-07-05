"""
Train BPRMF / NGCF / LightGCN x 3 seed (2026, 42, 123) trên ViFoodRec,
lưu predictions để longtail_analysis.py dùng tính Head/Body/Tail Recall@20.

Logic giữ nguyên:
  - BPRMF, NGCF : y hệt model.py / data.py / losses.py / evaluate.py trong
    lightgcn_baseline (không đổi gì cả).
  - LightGCN    : y hệt train_lightgcn_pyg.py
    (torch_geometric.nn.models.LightGCN, công thức Recall/NDCG copy nguyên
    từ gusye1234/LightGCN-PyTorch). Chỉ thêm early-stopping + lưu best
    checkpoint (giống 2 baseline kia) để 3 model so sánh công bằng.
  - Lưu predictions: dùng đúng logic save_predictions() của train.py
    (mask train items, top-K, lưu dict {user: np.array} vào .npy) cho cả
    3 model.

QUY ƯỚC PATH ĐÃ THỐNG NHẤT:
  - Checkpoint : checkpoints/{model}_th{threshold}_seed{seed}.pt
  - Predictions: outputs/predictions/{model}_th{threshold}_seed{seed}.npy
  - Data       : data/processed/experiments/th{threshold}/{train,test}.txt
  
Giả định (chỉnh lại ở phần CONFIG nếu khác):
  - Dataset dùng chung cho cả 3 model, threshold=3.5 mặc định (đổi biến
    THRESHOLD ở CONFIG nếu muốn chạy threshold khác) — để đảm bảo so sánh
    công bằng head/body/tail trên cùng 1 tập dữ liệu.
  - Hyperparam dùng chung cho cả 3 model: emb=64, layers=3, lr=1e-3,
    batch=1024, reg=1e-4, epochs=1000, eval_every=20, patience=10 — khớp
    Mục "Siêu tham số" trong báo cáo (đã xác nhận batch_size=1024 áp dụng
    cho cả BPR-MF/NGCF/LightGCN, không phải giá trị khác nhau giữa các model).

Cách dùng: đặt file này cùng cấp với config.py/data.py/model.py/losses.py/
evaluate.py (trong lightgcn_baseline), cài torch_geometric rồi chạy:
    pip install torch_geometric
    python run_all_baselines.py
"""

import os
import time
import random

import numpy as np
import torch

# ── modules gốc của repo — KHÔNG đổi logic BPRMF/NGCF ──
from data import InteractionData
from model import BPRMF, NGCF
from losses import bpr_loss
from evaluate import evaluate

from torch_geometric.nn.models import LightGCN as PyGLightGCN

# ============================================================
# CONFIG chung
# ============================================================
DATASET = "vifoodrec"
THRESHOLD = "3.5" 
_TH_DIR = f"th{THRESHOLD.replace('.', '_')}"
TRAIN_PATH = f"data/processed/experiments/{_TH_DIR}/train.txt"
TEST_PATH = f"data/processed/experiments/{_TH_DIR}/test.txt"
SEEDS = [2026, 42, 123]  
MODELS = ["bprmf", "ngcf", "lightgcn"]

EMB_DIM = 64
N_LAYERS = 3
DROPOUT = 0.1
LR = 0.001
REG_LAMBDA = 1e-4

BATCH_SIZE = 1024  # dùng chung cho cả 3 model (BPR-MF, NGCF, LightGCN)

EPOCHS = 1000
EVAL_EVERY = 20
EARLY_STOP_PATIENCE = 10
K = 20
EVAL_BATCH_SIZE = 1000

CKPT_DIR = "checkpoints"
PRED_DIR = "outputs/predictions"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) BPRMF / NGCF — GIỮ NGUYÊN logic train.py gốc
# ============================================================
def build_model_mf(name, data, norm_adj):
    if name == "bprmf":
        model = BPRMF(n_users=data.n_users, n_items=data.n_items, emb_dim=EMB_DIM)
    elif name == "ngcf":
        model = NGCF(
            n_users=data.n_users,
            n_items=data.n_items,
            emb_dim=EMB_DIM,
            n_layers=N_LAYERS,
            norm_adj=norm_adj,
            dropout=DROPOUT,
        )
    else:
        raise ValueError(name)
    return model.to(DEVICE)


@torch.no_grad()
def save_predictions_mf(model, data, k, save_path):
    """Y hệt save_predictions() trong train.py gốc."""
    model.eval()
    users_emb, items_emb = model.get_all_embeddings()
    test_users = [u for u in data.test_user_items if len(data.test_user_items[u]) > 0]

    predictions = {}
    batch_size = 1000
    for start in range(0, len(test_users), batch_size):
        batch = test_users[start : start + batch_size]
        u_emb = users_emb[torch.LongTensor(batch).to(DEVICE)]
        scores = u_emb @ items_emb.T
        for row, u in enumerate(batch):
            train_items = data.train_user_items.get(u, [])
            if train_items:
                scores[row, train_items] = -1e9
        _, top_k_idx = torch.topk(scores, k, dim=1)
        top_k_idx = top_k_idx.cpu().numpy()
        for row, u in enumerate(batch):
            predictions[u] = top_k_idx[row]

    np.save(save_path, predictions)
    print(f"[INFO] Predictions -> {save_path} ({len(predictions)} users)")


def train_mf_model(model_name, seed):
    print("=" * 70)
    print(f"[RUN] model={model_name.upper()} | seed={seed}")
    set_seed(seed)

    data = InteractionData(TRAIN_PATH, TEST_PATH)
    norm_adj = data.build_norm_adj().to(DEVICE)

    model = build_model_mf(model_name, data, norm_adj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    n_batches = max(1, data.n_train // BATCH_SIZE)
    ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_{_TH_DIR}_seed{seed}.pt")

    best_recall = best_ndcg = best_precision = 0.0
    best_epoch = no_improve = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos, neg = data.sample_batch(BATCH_SIZE)
            users, pos, neg = users.to(DEVICE), pos.to(DEVICE), neg.to(DEVICE)
            out = model(users, pos, neg)
            loss, _, _ = bpr_loss(*out, REG_LAMBDA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            recall, precision, ndcg = evaluate(
                model,
                data.train_user_items,
                data.test_user_items,
                data.n_items,
                k=K,
                batch_users=EVAL_BATCH_SIZE,
                device=DEVICE,
            )
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:4d} | loss {epoch_loss:.4f} | "
                f"recall@{K} {recall:.4f} | prec@{K} {precision:.4f} | "
                f"ndcg@{K} {ndcg:.4f} | {elapsed:.0f}s"
            )

            if recall > best_recall:
                best_recall, best_ndcg, best_precision = recall, ndcg, precision
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_recall": best_recall,
                        "best_ndcg": best_ndcg,
                        "best_precision": best_precision,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping tại epoch {epoch}")
                break

    print(
        f"[RESULT] {model_name.upper()} seed={seed} | best_epoch={best_epoch} | "
        f"recall@{K}={best_recall:.4f} | ndcg@{K}={best_ndcg:.4f}"
    )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    pred_path = os.path.join(PRED_DIR, f"{model_name}_{_TH_DIR}_seed{seed}.npy")
    save_predictions_mf(model, data, K, pred_path)


# ============================================================
# 2) LightGCN — GIỮ NGUYÊN logic train_lightgcn_pyg.py
# ============================================================
def load_interactions_lgn(path):
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


# ---- copy nguyên từ gusye1234/LightGCN-PyTorch/code/utils.py ----
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


# ---- hết phần copy ----


def train_lightgcn(seed):
    print("=" * 70)
    print(f"[RUN] model=LIGHTGCN (PyG) | seed={seed}")
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_pos, n_user_tr, n_item_tr = load_interactions_lgn(TRAIN_PATH)
    test_pos, n_user_te, n_item_te = load_interactions_lgn(TEST_PATH)
    num_users = max(n_user_tr, n_user_te)
    num_items = max(n_item_tr, n_item_te)
    num_nodes = num_users + num_items
    print(f"[Data] num_users={num_users}, num_items={num_items}, device={DEVICE}")

    train_u, train_i = [], []
    for u, items in train_pos.items():
        for it in items:
            train_u.append(u)
            train_i.append(it + num_users)
    train_u = np.array(train_u)
    train_i = np.array(train_i)
    edge_index = torch.tensor(
        np.stack(
            [np.concatenate([train_u, train_i]), np.concatenate([train_i, train_u])]
        ),
        dtype=torch.long,
        device=DEVICE,
    )
    train_data_size = len(train_u)
    print(
        f"[Data] {train_data_size} interactions for training, "
        f"{sum(len(v) for v in test_pos.values())} for testing"
    )

    model = PyGLightGCN(
        num_nodes=num_nodes, embedding_dim=EMB_DIM, num_layers=N_LAYERS
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def sample_bpr_batch():
        idx = np.random.randint(0, train_data_size, BATCH_SIZE)
        users = train_u[idx]
        pos_items = train_i[idx]
        neg_items = np.random.randint(num_users, num_users + num_items, BATCH_SIZE)
        return (
            torch.tensor(users, device=DEVICE),
            torch.tensor(pos_items, device=DEVICE),
            torch.tensor(neg_items, device=DEVICE),
        )

    @torch.no_grad()
    def evaluate_lgn():
        model.eval()
        out = model.get_embedding(edge_index)
        user_emb, item_emb = out[:num_users], out[num_users:]
        test_users = list(test_pos.keys())
        groundtruth, topk_items = [], []
        for u in test_users:
            scores = (user_emb[u] @ item_emb.T).clone()
            if u in train_pos:
                scores[train_pos[u]] = -1e10
            topk = torch.topk(scores, K).indices.cpu().numpy()
            topk_items.append(topk)
            groundtruth.append(test_pos[u])
        r = getLabel(groundtruth, topk_items)
        recall, precision = RecallPrecision_ATk(groundtruth, r, K)
        ndcg = NDCGatK_r(groundtruth, r, K)
        n = len(test_users)
        model.train()
        return recall / n, precision / n, ndcg / n

    n_batches = train_data_size // BATCH_SIZE + 1
    ckpt_path = os.path.join(CKPT_DIR, f"lightgcn_{_TH_DIR}_seed{seed}.pt")

    best_recall = best_ndcg = best_precision = 0.0
    best_epoch = no_improve = 0
    t0 = time.time()

    # Thêm early-stopping + lưu best checkpoint (đồng bộ với BPRMF/NGCF) —
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for _ in range(n_batches):
            users, pos, neg = sample_bpr_batch()
            edge_label_index = torch.stack(
                [torch.cat([users, users]), torch.cat([pos, neg])]
            )
            pos_rank, neg_rank = model(edge_index, edge_label_index).chunk(2)
            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
                lambda_reg=REG_LAMBDA,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / n_batches

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            recall, precision, ndcg = evaluate_lgn()
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:4d} | loss {epoch_loss:.4f} | "
                f"recall@{K} {recall:.4f} | prec@{K} {precision:.4f} | "
                f"ndcg@{K} {ndcg:.4f} | {elapsed:.0f}s"
            )

            if recall > best_recall:
                best_recall, best_ndcg, best_precision = recall, ndcg, precision
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_recall": best_recall,
                        "best_ndcg": best_ndcg,
                        "best_precision": best_precision,
                    },
                    ckpt_path,
                )
            else:
                no_improve += 1

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping tại epoch {epoch}")
                break

    print(
        f"[RESULT] LIGHTGCN seed={seed} | best_epoch={best_epoch} | "
        f"recall@{K}={best_recall:.4f} | ndcg@{K}={best_ndcg:.4f}"
    )

    # ── Lưu predictions — cùng logic save_predictions() của train.py ──
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        out = model.get_embedding(edge_index)
        user_emb, item_emb = out[:num_users], out[num_users:]

        predictions = {}
        test_users_all = [u for u in test_pos if len(test_pos[u]) > 0]
        batch_size = 1000
        for start in range(0, len(test_users_all), batch_size):
            batch = test_users_all[start : start + batch_size]
            u_emb = user_emb[torch.LongTensor(batch).to(DEVICE)]
            scores = u_emb @ item_emb.T
            for row, u in enumerate(batch):
                train_items = train_pos.get(u, [])
                if train_items:
                    scores[row, train_items] = -1e9
            _, top_k_idx = torch.topk(scores, K, dim=1)
            top_k_idx = top_k_idx.cpu().numpy()
            for row, u in enumerate(batch):
                predictions[u] = top_k_idx[row]

    pred_path = os.path.join(PRED_DIR, f"lightgcn_{_TH_DIR}_seed{seed}.npy")
    np.save(pred_path, predictions)
    print(f"[INFO] Predictions -> {pred_path} ({len(predictions)} users)")


# ============================================================
# MAIN — chạy 3 model x 3 seed
# ============================================================
def main():
    for model_name in MODELS:
        for seed in SEEDS:
            if model_name == "lightgcn":
                train_lightgcn(seed)
            else:
                train_mf_model(model_name, seed)

    print("\n[DONE] Đã train xong 3 model x 3 seed. Predictions nằm trong:")
    print(f"  {PRED_DIR}")
    print(
        "Chạy longtail_analysis.py (đã cập nhật path mới) để tính Head/Body/Tail Recall@20."
    )


if __name__ == "__main__":
    main()
