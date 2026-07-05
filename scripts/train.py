"""
train.py — Train BPR-MF / NGCF trên ViFoodRec.
Chọn model/threshold/seed qua CLI, vd:
    python train.py --model bprmf --threshold 3.5 --seed 42
(xem config.py để biết các cờ hỗ trợ)
"""

import os
import time

import numpy as np
import torch

from config import CONFIG
from data import InteractionData
from model import BPRMF, NGCF
from losses import bpr_loss
from evaluate import evaluate


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, data, norm_adj, device):
    name = cfg["model"]
    if name == "bprmf":
        model = BPRMF(
            n_users=data.n_users,
            n_items=data.n_items,
            emb_dim=cfg["emb_dim"],
        )
    elif name == "ngcf":
        model = NGCF(
            n_users=data.n_users,
            n_items=data.n_items,
            emb_dim=cfg["emb_dim"],
            n_layers=cfg["n_layers"],
            norm_adj=norm_adj,
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Chọn: bprmf | ngcf")
    return model.to(device)


def main():
    set_seed(CONFIG["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and CONFIG["device"] == "cuda" else "cpu"
    )
    print(
        f"[INFO] Model: {CONFIG['model'].upper()} | Dataset: {CONFIG['dataset']} | Device: {device}"
    )

    # ---- Load data ----
    data = InteractionData(CONFIG["train_path"], CONFIG["test_path"])

    # norm_adj không cần cho BPRMF nhưng build sẵn để code đồng nhất
    norm_adj = data.build_norm_adj().to(device)

    # ---- Build model ----
    model = build_model(CONFIG, data, norm_adj, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    n_batches = max(1, data.n_train // CONFIG["batch_size"])
    print(
        f"[INFO] n_users={data.n_users} | n_items={data.n_items} | "
        f"n_train={data.n_train} | n_batches/epoch={n_batches}"
    )

    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)

    # ---- Resume ----
    best_recall = best_ndcg = best_precision = 0.0
    best_epoch = no_improve = 0
    start_epoch = 1

    if CONFIG.get("resume") and os.path.exists(CONFIG["checkpoint_path"]):
        ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt["best_recall"]
        best_ndcg = ckpt["best_ndcg"]
        best_epoch = ckpt["epoch"]
        no_improve = ckpt.get("no_improve_count", 0)
        print(f"[INFO] Resume từ epoch {ckpt['epoch']} | best_recall={best_recall:.4f}")
    else:
        print("[INFO] Train từ đầu")

    print(
        f"[INFO] Epochs {start_epoch}→{CONFIG['epochs']}, "
        f"batch={CONFIG['batch_size']}, reg={CONFIG['reg_lambda']}"
    )

    k = CONFIG["k"]
    t0 = time.time()

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            users, pos, neg = data.sample_batch(CONFIG["batch_size"])
            users, pos, neg = users.to(device), pos.to(device), neg.to(device)
            out = model(users, pos, neg)
            loss, _, _ = bpr_loss(*out, CONFIG["reg_lambda"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        if epoch % CONFIG["eval_every"] == 0 or epoch == CONFIG["epochs"]:
            recall, precision, ndcg = evaluate(
                model,
                data.train_user_items,
                data.test_user_items,
                data.n_items,
                k=k,
                batch_users=CONFIG["eval_batch_size"],
                device=device,
            )
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:4d} | loss {epoch_loss:.4f} | "
                f"recall@{k} {recall:.4f} | prec@{k} {precision:.4f} | "
                f"ndcg@{k} {ndcg:.4f} | {elapsed:.0f}s"
            )

            if recall > best_recall:
                best_recall, best_ndcg, best_precision = recall, ndcg, precision
                best_epoch = epoch
                no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_recall": best_recall,
                        "best_ndcg": best_ndcg,
                        "best_precision": best_precision,
                        "no_improve_count": no_improve,
                    },
                    CONFIG["checkpoint_path"],
                )
            else:
                no_improve += 1

            if no_improve >= CONFIG["early_stop_patience"]:
                print(f"[INFO] Early stopping tại epoch {epoch}")
                break

    print("=" * 60)
    print(f"[RESULT] Model:           {CONFIG['model'].upper()}")
    print(f"[RESULT] Best epoch:      {best_epoch}")
    print(f"[RESULT] Recall@{k}:  {best_recall:.4f}")
    print(f"[RESULT] Precision@{k}:{best_precision:.4f}")
    print(f"[RESULT] NDCG@{k}:    {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
