"""
Train LightGCN tren Gowalla dung torch_geometric.nn.models.LightGCN
-> Cung thu vien voi ViFoodRec, ket qua co the so sanh truc tiep trong bao cao.

Muc tieu:
    Tai hien ket qua paper goc (He et al., SIGIR 2020) bang thu vien torch_geometric
    de chung minh hai pipeline (Gowalla va ViFoodRec) su dung cung cai dat.
    Ket qua mong doi: Recall@20 ~0.183, NDCG@20 ~0.156, Precision@20 ~0.056

Data can:
    data/gowalla/train.txt  )  tai tu:
    data/gowalla/test.txt   )  github.com/gusye1234/LightGCN-PyTorch/tree/master/data/gowalla
    (cung format voi ViFoodRec: "<user_idx> <item1> <item2> ...")

Luu y ve thoi gian:
    Gowalla: 29,858 users, 40,981 items, 810,128 interactions
    -> Uoc tinh ~35-50s/epoch tren Kaggle T4 (lau hon ViFoodRec ~30-40 lan)
    -> 1000 epoch ~ 10-14 gio, phu hop voi "Save & Run All" cua Kaggle (gioi han 12h)
    -> Neu bi cat truoc epoch 1000: checkpoint cuoi duoc luu tu dong,
       co the evaluate bang evaluate_checkpoint.py

Output:
    lightgcn_gowalla_seed{SEED}.pt
    history_gowalla_seed{SEED}.json   (loss + metric moi TEST_EVERY epoch)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn.models import LightGCN

# ===================== CONFIG =====================
DATA_DIR = "data/gowalla"
EMBEDDING_DIM = 64
N_LAYERS = 3
LR = 0.001
DECAY = 1e-4
BATCH_SIZE = 2048  # lon hon ViFoodRec vi dataset lon hon nhieu
N_EPOCHS = 1000
TOPK = 20
TEST_EVERY = 20  # Gowalla: evaluate it hon de tiet kiem thoi gian
SEED = 2026
# ==================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


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


# ── Metric: copy nguyen tu gusye1234/LightGCN-PyTorch/code/utils.py ──
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


# ─────────────────────────────────────────────────────────────────────


def main():
    # ── Load data ──
    train_pos, nu_tr, ni_tr = load_interactions(Path(DATA_DIR) / "train.txt")
    test_pos, nu_te, ni_te = load_interactions(Path(DATA_DIR) / "test.txt")
    num_users = max(nu_tr, nu_te)
    num_items = max(ni_tr, ni_te)
    num_nodes = num_users + num_items
    print(
        f"[Data] users={num_users}, items={num_items}, "
        f"train_interactions={sum(len(v) for v in train_pos.values())}, device={DEVICE}"
    )
    print(
        f"[Config] emb={EMBEDDING_DIM}, layers={N_LAYERS}, lr={LR}, "
        f"decay={DECAY}, batch={BATCH_SIZE}, epochs={N_EPOCHS}"
    )

    # ── Build edge_index (item id offset boi num_users) ──
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

    # ── Model ──
    model = LightGCN(
        num_nodes=num_nodes, embedding_dim=EMBEDDING_DIM, num_layers=N_LAYERS
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"[Model] {model}")

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
    def evaluate():
        model.eval()
        out = model.get_embedding(edge_index)
        user_emb = out[:num_users]
        item_emb = out[num_users:]

        # Gowalla lon: chia batch de tranh OOM
        test_users = list(test_pos.keys())
        groundtruth, topk_items = [], []
        batch_size_eval = 512
        for start in range(0, len(test_users), batch_size_eval):
            batch_u = test_users[start : start + batch_size_eval]
            scores = user_emb[batch_u] @ item_emb.T  # (B, n_items)
            # Mask train items
            for i, u in enumerate(batch_u):
                if u in train_pos:
                    scores[i, train_pos[u]] = -1e10
            topk = torch.topk(scores, TOPK, dim=1).indices.cpu().numpy()
            for i, u in enumerate(batch_u):
                topk_items.append(topk[i].tolist())
                groundtruth.append(test_pos[u])

        r = getLabel(groundtruth, topk_items)
        recall, precision = RecallPrecision_ATk(groundtruth, r, TOPK)
        ndcg = NDCGatK_r(groundtruth, r, TOPK)
        n = len(test_users)
        model.train()
        return recall / n, precision / n, ndcg / n

    # ── Train loop ──
    n_batches = train_data_size // BATCH_SIZE + 1
    history = []
    label = Path(DATA_DIR).name  # "gowalla"

    for epoch in range(N_EPOCHS):
        start = time.time()
        total_loss = 0.0

        for _ in range(n_batches):
            users, pos, neg = sample_bpr_batch()
            eli = torch.stack([torch.cat([users, users]), torch.cat([pos, neg])])
            pos_rank, neg_rank = model(edge_index, eli).chunk(2)
            loss = model.recommendation_loss(
                pos_rank, neg_rank, node_id=eli.unique(), lambda_reg=DECAY
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start
        avg_loss = total_loss / n_batches
        log_line = (
            f"Epoch[{epoch+1}/{N_EPOCHS}] loss={avg_loss:.4f} "
            f"time={epoch_time:.2f}s"
        )
        epoch_record = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "epoch_time_sec": round(epoch_time, 2),
        }

        if epoch % TEST_EVERY == 0 or epoch == N_EPOCHS - 1:
            recall, precision, ndcg = evaluate()
            log_line += (
                f" | [TEST] Recall@{TOPK}={recall:.4f} "
                f"Precision@{TOPK}={precision:.4f} "
                f"NDCG@{TOPK}={ndcg:.4f}"
            )
            epoch_record.update(
                {
                    "recall": round(recall, 6),
                    "precision": round(precision, 6),
                    "ndcg": round(ndcg, 6),
                }
            )

        history.append(epoch_record)
        print(log_line)

        # Luu checkpoint dinh ky de phong mat ket noi
        if (epoch + 1) % 100 == 0:
            ckpt_periodic = f"lightgcn_{label}_seed{SEED}_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_periodic)
            print(f"  [Checkpoint] Da luu: {ckpt_periodic}")

    # ── Luu ket qua cuoi ──
    ckpt_final = f"lightgcn_{label}_seed{SEED}.pt"
    torch.save(model.state_dict(), ckpt_final)
    print(f"\nDa luu checkpoint cuoi: {ckpt_final}")

    history_path = f"history_{label}_seed{SEED}.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "dataset": label,
                "seed": SEED,
                "topk": TOPK,
                "config": {
                    "embedding_dim": EMBEDDING_DIM,
                    "n_layers": N_LAYERS,
                    "lr": LR,
                    "decay": DECAY,
                    "batch_size": BATCH_SIZE,
                    "n_epochs": N_EPOCHS,
                },
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Da luu history: {history_path}")

    # ── In ket qua cuoi de doi chieu voi paper ──
    last_test = next((r for r in reversed(history) if "recall" in r), None)
    if last_test:
        print(f"\n{'='*55}")
        print(f"  KET QUA CUOI — LightGCN tren Gowalla (seed={SEED})")
        print(f"{'='*55}")
        print(f"  Recall@{TOPK}    = {last_test['recall']:.4f}  (paper: 0.1830)")
        print(f"  Precision@{TOPK} = {last_test['precision']:.4f}  (paper: 0.0561)")
        print(f"  NDCG@{TOPK}      = {last_test['ndcg']:.4f}  (paper: 0.1557)")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
