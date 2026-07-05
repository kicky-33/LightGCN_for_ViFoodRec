"""
model.py — 3 models cho comparison:
  1. BPRMF    — Matrix Factorization + BPR (baseline đơn giản nhất)
  2. NGCF     — Neural Graph CF (Wang et al., SIGIR 2019)

Tất cả dùng chung: BPR loss, evaluate(), InteractionData, train loop.
Chọn model bằng cờ --model khi chạy train.py (xem config.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. BPR-MF
# ============================================================
class BPRMF(nn.Module):
    """
    Matrix Factorization tối ưu bằng BPR loss.
    Không có graph propagation — baseline đơn giản nhất.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, **kwargs):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, pos_items, neg_items):
        u = self.user_emb(users)
        pos_i = self.item_emb(pos_items)
        neg_i = self.item_emb(neg_items)
        # MF: ego = final embedding (không có propagation)
        return u, pos_i, neg_i, u, pos_i, neg_i

    @torch.no_grad()
    def get_all_embeddings(self):
        self.eval()
        return self.user_emb.weight, self.item_emb.weight


# ============================================================
# 2. NGCF
# ============================================================
class NGCF(nn.Module):
    """
    NGCF: Wang et al., SIGIR 2019.
    Khác LightGCN:
      - W1, W2 feature transformation mỗi layer
      - LeakyReLU activation
      - Interaction term (e_neighbor ⊙ e_self) trong message
      - Self-connection
      - Final = concat tất cả layers (không phải mean)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_layers: int = 3,
        norm_adj: torch.Tensor = None,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj
        self.dropout = dropout

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.W1 = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)]
        )
        self.W2 = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim, bias=False) for _ in range(n_layers)]
        )
        for w in list(self.W1) + list(self.W2):
            nn.init.xavier_uniform_(w.weight)

        self.leaky = nn.LeakyReLU(0.2)

    def propagate(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]

        for k in range(self.n_layers):
            # neighbor aggregation: Ã * E^(k)
            neighbor_agg = torch.sparse.mm(self.norm_adj, all_emb)

            # interaction term đúng theo paper: (Ã * E^(k)) ⊙ E^(k)
            # tức là e_neighbor ⊙ e_self sau khi aggregate
            interact = neighbor_agg * all_emb

            # self-connection + neighbor message + interaction
            msg = self.W1[k](neighbor_agg) + self.W2[k](interact) + self.W1[k](all_emb)

            if self.training and self.dropout > 0:
                msg = F.dropout(msg, p=self.dropout)

            all_emb = self.leaky(msg)
            embs.append(all_emb)

        # NGCF dùng concat (không phải mean như LightGCN)
        final_emb = torch.cat(embs, dim=1)
        users_final, items_final = torch.split(final_emb, [self.n_users, self.n_items])
        return users_final, items_final

    def forward(self, users, pos_items, neg_items):
        users_emb, items_emb = self.propagate()
        u = users_emb[users]
        pos_i = items_emb[pos_items]
        neg_i = items_emb[neg_items]
        u_ego = self.user_emb(users)
        pos_ego = self.item_emb(pos_items)
        neg_ego = self.item_emb(neg_items)
        return u, pos_i, neg_i, u_ego, pos_ego, neg_ego

    @torch.no_grad()
    def get_all_embeddings(self):
        self.eval()
        return self.propagate()
