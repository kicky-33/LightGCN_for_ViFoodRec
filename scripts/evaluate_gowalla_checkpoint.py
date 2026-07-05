"""
evaluate_gowalla_checkpoint.py
================================
Đánh giá lại checkpoint Gowalla ĐÃ TRAIN SẴN bằng đúng code gốc của tác giả
(gusye1234/LightGCN-PyTorch), KHÔNG train lại — dùng để xác nhận số liệu
tái hiện baseline (Bảng "Tái hiện Baseline Gowalla" trong báo cáo) mà không
tốn thời gian train lại từ đầu.

QUAN TRỌNG VỀ KIẾN TRÚC:
    Checkpoint "lgn-gowalla-3-64.pth.tar" được train bằng code GỐC của
    gusye1234 (không phải torch_geometric.nn.models.LightGCN đang dùng cho
    phần ViFoodRec). State_dict của checkpoint này có 2 tensor riêng:
        embedding_user.weight  (n_users, latent_dim)
        embedding_item.weight  (n_items, latent_dim)
    khác với torch_geometric.nn.models.LightGCN vốn gộp chung thành
    1 tensor "embedding.weight" (n_users+n_items, latent_dim).
    => Không thể dùng chung evaluate_checkpoint.py (torch_geometric) để load
       checkpoint này — file này định nghĩa lại đúng kiến trúc gốc
       (embedding_user/embedding_item + mean-pooling qua các layer) chỉ để
       LOAD WEIGHT và ĐÁNH GIÁ, không có hàm train.

Công thức Recall@K/Precision@K/NDCG@K và chuẩn hóa đồ thị (D^-1/2 A D^-1/2)
được copy nguyên từ gusye1234/LightGCN-PyTorch/code/{utils.py, dataloader.py}
để đảm bảo tái lập đúng số liệu đã báo cáo.

Cách dùng:
    python scripts/evaluate_gowalla_checkpoint.py \
        --ckpt checkpoints/gowalla/lgn-gowalla-3-64.pth.tar \
        --data-dir data/gowalla \
        --n-layers 3 --emb-dim 64 --topk 20

Output:
    In ra console Recall@20/Precision@20/NDCG@20 + % sai lệch so với paper gốc
    (He et al., 2020), đồng thời ghi outputs/results/gowalla_reproduction.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

# Số liệu Recall@20/NDCG@20/Precision@20 báo cáo trong paper gốc (He et al., 2020, SIGIR)
PAPER_GOWALLA = {"recall": 0.1830, "ndcg": 0.1557, "precision": 0.0561}


def parse_args():
    p = argparse.ArgumentParser(description="Đánh giá checkpoint Gowalla gốc (gusye1234)")
    p.add_argument("--ckpt", required=True, help="Path tới file .pth.tar (vd: checkpoints/gowalla/lgn-gowalla-3-64.pth.tar)")
    p.add_argument("--data-dir", default="data/gowalla", help="Thư mục chứa train.txt/test.txt dạng Gowalla")
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out-csv", default="outputs/results/gowalla_reproduction.csv")
    return p.parse_args()


# ---------- Load data (format LightGCN chuẩn: "user item1 item2 ...") ----------
def load_interactions(path):
    data, n_user, n_item = {}, 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(" ")
            u = int(parts[0])
            items = [int(x) for x in parts[1:] if x != ""]
            if not items:
                continue
            data[u] = items
            n_user = max(n_user, u)
            n_item = max(n_item, max(items))
    return data, n_user + 1, n_item + 1


def build_norm_adj(train_user_items, n_users, n_items):
    """A = [[0,R],[R^T,0]], A_tilde = D^-1/2 A D^-1/2 — đúng công thức (6)-(7)
    trong paper LightGCN, giống getSparseGraph() của gusye1234."""
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for u, items in train_user_items.items():
        for i in items:
            R[u, i] = 1.0
    R = R.tocsr()

    zero_uu = sp.csr_matrix((n_users, n_users))
    zero_ii = sp.csr_matrix((n_items, n_items))
    A = sp.vstack([sp.hstack([zero_uu, R]), sp.hstack([R.T, zero_ii])]).tocsr()

    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = np.power(deg[nz], -0.5)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)

    A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()
    indices = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
    values = torch.FloatTensor(A_norm.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(A_norm.shape)).coalesce()


# ---------- Model — đúng kiến trúc gốc gusye1234 (embedding_user/embedding_item riêng) ----------
class OriginalLightGCN(nn.Module):
    """
    Tái hiện tối giản class LightGCN trong gusye1234/LightGCN-PyTorch/code/model.py,
    CHỈ đủ để load state_dict checkpoint gốc và forward — không có phần train.
    State_dict keys bắt buộc phải khớp: "embedding_user.weight", "embedding_item.weight".
    """

    def __init__(self, n_users, n_items, emb_dim, n_layers, norm_adj):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.embedding_user = nn.Embedding(n_users, emb_dim)
        self.embedding_item = nn.Embedding(n_items, emb_dim)

    @torch.no_grad()
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        light_out = torch.stack(embs, dim=1).mean(dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items


# ---------- Metric — COPY NGUYEN tu gusye1234/LightGCN-PyTorch/code/utils.py ----------
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
        pred = list(map(lambda x: x in test_data[i], pred_data[i]))
        r.append(np.array(pred).astype("float"))
    return np.array(r)
# ---------- het phan copy tu gusye1234 ----------


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {ckpt_path}\n"
            f"Kiểm tra lại --ckpt (vd: checkpoints/gowalla/lgn-gowalla-3-64.pth.tar)"
        )

    print(f"[Data] Đọc train/test từ: {data_dir}")
    train_pos, n_user_tr, n_item_tr = load_interactions(data_dir / "train.txt")
    test_pos, n_user_te, n_item_te = load_interactions(data_dir / "test.txt")
    n_users = max(n_user_tr, n_user_te)
    n_items = max(n_item_tr, n_item_te)
    print(f"[Data] n_users={n_users}, n_items={n_items}, device={device}")

    norm_adj = build_norm_adj(train_pos, n_users, n_items).to(device)

    model = OriginalLightGCN(
        n_users=n_users, n_items=n_items,
        emb_dim=args.emb_dim, n_layers=args.n_layers, norm_adj=norm_adj,
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[!] Cảnh báo load_state_dict — missing={missing}, unexpected={unexpected}")
        print("    Kiểm tra lại --emb-dim/--n-layers có khớp checkpoint không "
              "(tên file 'lgn-gowalla-3-64' nghĩa là n_layers=3, emb_dim=64).")
    model.eval()

    with torch.no_grad():
        user_emb, item_emb = model.computer()
        test_users = list(test_pos.keys())
        groundtruth, topk_items = [], []
        for u in test_users:
            scores = (user_emb[u] @ item_emb.T).clone()
            if u in train_pos:
                scores[train_pos[u]] = -1e10  # loại item đã thấy ở train
            topk = torch.topk(scores, args.topk).indices.cpu().numpy()
            topk_items.append(topk)
            groundtruth.append(test_pos[u])

    r = getLabel(groundtruth, topk_items)
    recall, precision = RecallPrecision_ATk(groundtruth, r, args.topk)
    ndcg = NDCGatK_r(groundtruth, r, args.topk)
    n = len(test_users)
    recall, precision, ndcg = recall / n, precision / n, ndcg / n

    print(f"\n{'='*60}")
    print(f"  TÁI HIỆN BASELINE GOWALLA — checkpoint: {ckpt_path.name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<12}{'Nhóm đạt':>12}{'Paper gốc':>12}{'Sai lệch':>12}")
    for name, val in [("Recall@20", recall), ("NDCG@20", ndcg), ("Precision@20", precision)]:
        key = name.split("@")[0].lower()
        paper_val = PAPER_GOWALLA[key]
        diff_pct = (val - paper_val) / paper_val * 100
        print(f"  {name:<12}{val:>12.4f}{paper_val:>12.4f}{diff_pct:>11.2f}%")
    print(f"{'='*60}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "reproduced", "paper", "diff_pct"])
        for name, val in [("recall@20", recall), ("ndcg@20", ndcg), ("precision@20", precision)]:
            key = name.split("@")[0]
            paper_val = PAPER_GOWALLA[key]
            diff_pct = (val - paper_val) / paper_val * 100
            writer.writerow([name, round(val, 6), paper_val, round(diff_pct, 4)])
    print(f"\nĐã ghi: {out_path}")


if __name__ == "__main__":
    main()
