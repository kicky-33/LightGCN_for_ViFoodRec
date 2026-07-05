"""
config.py — Cấu hình dùng chung cho BPR-MF / NGCF trên ViFoodRec.

Có thể override qua CLI, ví dụ:
    python train.py --model bprmf --threshold 3.5 --seed 42
    python train.py --model lightgcn --threshold 4.0 --seed 2026

Quy ước checkpoint thống nhất toàn repo:
    checkpoints/{model}_th{threshold}_seed{seed}.pt
    (không tiền tố "best_", không path tuyệt đối Kaggle — path luôn tương đối so
    với thư mục gốc repo để chạy được trên mọi máy/mọi môi trường)
"""

import argparse


def _threshold_to_dirname(t: str) -> str:
    """'3.5' -> 'th3_5')"""
    return f"th{t.replace('.', '_')}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BPR-MF / NGCF / LightGCN trên ViFoodRec"
    )
    parser.add_argument("--model", choices=["bprmf", "ngcf"], default="bprmf")
    parser.add_argument(
        "--threshold",
        default="3.5",
        help="Ngưỡng binarize đã tiền xử lý, vd: 3.0, 3.5, 4.0, 4.5",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def build_config(model: str, threshold: str, seed: int, resume: bool = True) -> dict:
    th_dir = _threshold_to_dirname(threshold)
    return {
        "model": model,
        "dataset": "vifoodrec",
        "threshold": threshold,
        "seed": seed,
        "train_path": f"data/processed/experiments/{th_dir}/train.txt",
        "test_path": f"data/processed/experiments/{th_dir}/test.txt",
        "checkpoint_path": f"checkpoints/{model}_{th_dir}_seed{seed}.pt",
        # Model
        "emb_dim": 64,
        "n_layers": 3,  # dùng cho LightGCN và NGCF; bỏ qua với BPRMF
        "dropout": 0.1,  # chỉ dùng cho NGCF
        # Optimization
        "lr": 0.001,
        # batch_size = 1024 dùng chung cho cả BPR-MF / NGCF / LightGCN
        # (khớp Mục "Siêu tham số" trong báo cáo — đã xác nhận đây là giá trị đúng,
        # không phải 4096 như bản draft trước đó của config.py).
        "batch_size": 1024,
        "reg_lambda": 1e-4,  # NGCF paper dùng 1e-4; LightGCN Yelp dùng 1e-3 (không áp dụng ở đây)
        # Training
        "epochs": 1000,
        "eval_every": 20,
        "early_stop_patience": 10,
        # Evaluation
        "k": 20,
        "eval_batch_size": 1000,
        "device": "cuda",
        "resume": resume,
    }


_args = None
try:
    _args = parse_args()
except SystemExit:
    _args = argparse.Namespace(model="bprmf", threshold="3.5", seed=2026, resume=True)

CONFIG = build_config(_args.model, _args.threshold, _args.seed, _args.resume)
