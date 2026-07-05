"""
Train LightGCN tren ViFoodRec dung THU VIEN torch_geometric.nn.models.LightGCN
-> KHONG can clone/patch repo gusye1234/LightGCN-PyTorch.

Cai dat:
    pip install torch_geometric

QUAN TRONG VE TINH NHAT QUAN:
    Cong thuc RecallPrecision_ATk / NDCGatK_r / getLabel ben duoi duoc COPY NGUYEN
    tu gusye1234/LightGCN-PyTorch/code/utils.py (repo da dung de tai hien baseline
    Gowalla) -> dam bao Recall@20/NDCG@20/Precision@20 tinh ra co the so sanh truc
    tiep voi so lieu baseline Gowalla da co, du model duoc khoi tao bang thu vien khac.

Cau hinh mac dinh:
    EMBEDDING_DIM=64, N_LAYERS=3, LR=0.001, DECAY=1e-4, BATCH_SIZE=1024
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn.models import LightGCN

_parser = argparse.ArgumentParser()
_parser.add_argument("--data-dir", default="data/processed/experiments/th3_5")
_parser.add_argument("--seed", type=int, default=42)
_cli, _ = _parser.parse_known_args()

# ===================== CONFIG =====================
DATA_DIR = _cli.data_dir  # thu muc chua train.txt/test.txt CUA 1 THRESHOLD
EMBEDDING_DIM = 64
N_LAYERS = 3
LR = 0.001
DECAY = 1e-4  # lambda_reg cho BPR loss (= L2 reg)
BATCH_SIZE = 1024
N_EPOCHS = 1000
TOPK = 20
TEST_EVERY = 10  # danh gia tren test set moi X epoch
SEED = _cli.seed
# ==================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_interactions(path):
    """Doc file dang LightGCN format: '<user_idx> <item_idx1> <item_idx2> ...'
    Tra ve dict {user_idx: [item_idx, ...]}, cung n_user/n_item suy ra tu file."""
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


# ---------- Metric - COPY NGUYEN cong thuc tu gusye1234/LightGCN-PyTorch/code/utils.py ----------
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


# ---------- het phan copy tu gusye1234 ----------


def main():
    data_dir = Path(DATA_DIR)
    train_pos, n_user_tr, n_item_tr = load_interactions(data_dir / "train.txt")
    test_pos, n_user_te, n_item_te = load_interactions(data_dir / "test.txt")
    num_users = max(n_user_tr, n_user_te)
    num_items = max(n_item_tr, n_item_te)
    num_nodes = num_users + num_items
    print(f"[Data] num_users={num_users}, num_items={num_items}, device={DEVICE}")

    # ---------- Build edge_index 2 chieu, item id offset boi num_users (chuan PyG) ----------
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

    model = LightGCN(
        num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=N_LAYERS
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def sample_bpr_batch():
        """BPR negative sampling - cung kieu uniform random nhu UniformSample_original_python
        cua gusye1234 (khong loai tru trung lap voi positive, dung chuan thuc hanh pho bien).
        """
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
    def evaluate():
        model.eval()
        out = model.get_embedding(edge_index)
        user_emb, item_emb = out[:num_users], out[num_users:]
        test_users = list(test_pos.keys())
        groundtruth, topk_items = [], []
        for u in test_users:
            scores = (user_emb[u] @ item_emb.T).clone()
            if u in train_pos:
                scores[train_pos[u]] = (
                    -1e10
                )  # loai item da thay o train, giong code goc
            topk = torch.topk(scores, TOPK).indices.cpu().numpy()
            topk_items.append(topk)
            groundtruth.append(test_pos[u])
        r = getLabel(groundtruth, topk_items)
        recall, precision = RecallPrecision_ATk(groundtruth, r, TOPK)
        ndcg = NDCGatK_r(groundtruth, r, TOPK)
        n = len(test_users)
        model.train()
        return recall / n, precision / n, ndcg / n

    n_batches = train_data_size // BATCH_SIZE + 1
    history = []
    for epoch in range(N_EPOCHS):
        start = time.time()
        total_loss = 0.0
        for _ in range(n_batches):
            users, pos, neg = sample_bpr_batch()
            edge_label_index = torch.stack(
                [torch.cat([users, users]), torch.cat([pos, neg])]
            )
            pos_rank, neg_rank = model(edge_index, edge_label_index).chunk(2)
            loss = model.recommendation_loss(
                pos_rank, neg_rank, node_id=edge_label_index.unique(), lambda_reg=DECAY
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_time = time.time() - start

        log_line = f"Epoch[{epoch + 1}/{N_EPOCHS}] loss={total_loss / n_batches:.4f} time={epoch_time:.2f}s"
        if epoch % TEST_EVERY == 0 or epoch == N_EPOCHS - 1:
            recall, precision, ndcg = evaluate()
            log_line += f" | [TEST] Recall@{TOPK}={recall:.4f} Precision@{TOPK}={precision:.4f} NDCG@{TOPK}={ndcg:.4f}"
            history.append(
                {
                    "epoch": epoch,
                    "recall": recall,
                    "precision": precision,
                    "ndcg": ndcg,
                    "loss": total_loss / n_batches,
                    "epoch_time_sec": epoch_time,
                }
            )
        print(log_line)

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = ckpt_dir / f"lightgcn_{Path(DATA_DIR).name}_seed{SEED}.pt"
    torch.save(model.state_dict(), ckpt_name)
    print(f"Da luu checkpoint: {ckpt_name}")
    return history


if __name__ == "__main__":
    main()
