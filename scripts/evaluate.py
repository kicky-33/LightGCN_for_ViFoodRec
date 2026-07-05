"""
evaluate.py — Recall@K, Precision@K, NDCG@K theo all-ranking protocol.
"""

import numpy as np
import torch


@torch.no_grad()
def evaluate(
    model,
    train_user_items: dict,
    test_user_items: dict,
    n_items: int,
    k: int = 20,
    batch_users: int = 1000,
    device="cpu",
):
    model.eval()
    users_emb, items_emb = model.get_all_embeddings()

    recalls, precisions, ndcgs = [], [], []
    test_users = [u for u in test_user_items if len(test_user_items[u]) > 0]
    log_idx = np.log2(np.arange(2, k + 2))

    for start in range(0, len(test_users), batch_users):
        batch = test_users[start : start + batch_users]
        u_emb = users_emb[torch.LongTensor(batch).to(device)]
        scores = u_emb @ items_emb.T

        # Mask train items
        for row, u in enumerate(batch):
            train_items = train_user_items.get(u, [])
            if train_items:
                scores[row, train_items] = -1e9

        _, top_k_idx = torch.topk(scores, k, dim=1)
        top_k_idx = top_k_idx.cpu().numpy()

        for row, u in enumerate(batch):
            gt = set(test_user_items[u])
            pred = top_k_idx[row]
            hits = np.isin(pred, list(gt)).astype(np.float32)

            recalls.append(hits.sum() / len(gt))
            precisions.append(hits.sum() / k)

            dcg = (hits / log_idx).sum()
            idcg = (1.0 / log_idx[: min(len(gt), k)]).sum()
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(precisions)), float(np.mean(ndcgs))
