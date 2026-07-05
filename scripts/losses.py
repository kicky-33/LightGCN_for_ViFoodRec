"""
losses.py — Bayesian Personalized Ranking (BPR) loss, công thức (15) trong paper LightGCN.
"""

import torch
import torch.nn.functional as F


def bpr_loss(u, pos_i, neg_i, u_ego, pos_ego, neg_ego, reg_lambda: float):
    pos_scores = (u * pos_i).sum(dim=1)
    neg_scores = (u * neg_i).sum(dim=1)

    mf_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    reg_loss = (
        u_ego.norm(2).pow(2) + pos_ego.norm(2).pow(2) + neg_ego.norm(2).pow(2)
    ) / u_ego.shape[0]

    return mf_loss + reg_lambda * reg_loss, mf_loss.item(), reg_loss.item()
