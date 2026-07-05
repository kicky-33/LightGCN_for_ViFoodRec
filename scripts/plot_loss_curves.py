"""
Parse log console Kaggle -> ve 4 bieu do Loss / Recall / NDCG theo epoch
(1 hinh moi threshold), xuat ra 4 file PNG cho bao cao.

CACH DUNG:
    1. Copy toan bo output cell Kaggle cua moi threshold vao file .txt tuong ung:
           logs/log_th3_0.txt
           logs/log_th3_5.txt
           logs/log_th4_0.txt
           logs/log_th4_5.txt

       Paste nguyen si, ke ca so thu tu dong va timestamp ben trai (Kaggle them vao),
       script se tu bo qua. Format moi dong co the la:
           1336.7s   1013   Epoch[1000/1000] loss=0.0621 time=1.13s | [TEST] Recall@20=0.0051 ...
       hoac don gian hon:
           Epoch[1000/1000] loss=0.0621 time=1.13s | [TEST] Recall@20=0.0051 ...
       Deu duoc.

    2. Chay:
           python plot_loss_curves.py

    3. Output: outputs/figures/loss_curve_th{X_X}.png x4

GHI CHU: moi log file chi can CUNG 1 threshold (1 lan train). Neu th3_5 train
3 seed, dung log cua seed chinh (2026) de ve bieu do — cac seed khac
chi dung de tinh mean+-std, khong can ve rieng.
"""

import re
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===================== CONFIG =====================
LOG_FILES = {
    "3.0": "logs/log_th3_0.txt",
    "3.5": "logs/log_th3_5.txt",
    "4.0": "logs/log_th4_0.txt",
    "4.5": "logs/log_th4_5.txt",
}
OUT_DIR = Path("outputs/figures")
TOPK = 20
# ==================================================

# Regex khop ca 2 dang dong: co timestamp Kaggle hoac khong
RE_EPOCH = re.compile(
    r"Epoch\[(\d+)/\d+\]\s+loss=([\d.]+)"
    r"(?:.*?\[TEST\]\s+Recall@\d+=(?P<recall>[\d.]+)"
    r"\s+Precision@\d+=(?P<precision>[\d.]+)"
    r"\s+NDCG@\d+=(?P<ndcg>[\d.]+))?"
)


def parse_log(path):
    epochs, losses, test_epochs, recalls, ndcgs = [], [], [], [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            loss = float(m.group(2))
            epochs.append(ep)
            losses.append(loss)
            if m.group("recall"):
                test_epochs.append(ep)
                recalls.append(float(m.group("recall")))
                ndcgs.append(float(m.group("ndcg")))
    return epochs, losses, test_epochs, recalls, ndcgs


def plot_one(
    ax_loss, ax_metric, threshold, epochs, losses, test_epochs, recalls, ndcgs
):
    color_loss = "#378ADD"
    color_recall = "#1D9E75"
    color_ndcg = "#D85A30"

    ax_loss.plot(epochs, losses, color=color_loss, linewidth=1.2, alpha=0.85)
    ax_loss.set_ylabel("BPR Loss", fontsize=9, color=color_loss)
    ax_loss.tick_params(axis="y", labelcolor=color_loss, labelsize=8)
    ax_loss.tick_params(axis="x", labelsize=8)
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_title(f"Threshold = {threshold}", fontsize=10, fontweight="bold", pad=6)
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax_loss.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    if test_epochs:
        ax_metric.plot(
            test_epochs,
            recalls,
            color=color_recall,
            linewidth=1.3,
            marker="o",
            markersize=2.5,
            label=f"Recall@{TOPK}",
        )
        ax_metric.plot(
            test_epochs,
            ndcgs,
            color=color_ndcg,
            linewidth=1.3,
            marker="s",
            markersize=2.5,
            linestyle="--",
            label=f"NDCG@{TOPK}",
        )
        ax_metric.set_ylabel("Metric", fontsize=9)
        ax_metric.tick_params(axis="y", labelsize=8)
        ax_metric.tick_params(axis="x", labelsize=8)
        ax_metric.legend(fontsize=7, loc="lower right", framealpha=0.6)
        ax_metric.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        # Danh dau gia tri tot nhat
        best_idx = recalls.index(max(recalls))
        ax_metric.axvline(
            test_epochs[best_idx],
            color=color_recall,
            linestyle=":",
            linewidth=0.8,
            alpha=0.7,
        )
        ax_metric.annotate(
            f"best Recall\n@ep{test_epochs[best_idx]}\n={max(recalls):.4f}",
            xy=(test_epochs[best_idx], max(recalls)),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=6.5,
            color=color_recall,
            arrowprops=dict(arrowstyle="->", color=color_recall, lw=0.8),
        )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for threshold, log_path in LOG_FILES.items():
        p = Path(log_path)
        if not p.exists():
            print(f"[!] Khong tim thay {log_path} — bo qua threshold {threshold}.")
            continue

        epochs, losses, test_epochs, recalls, ndcgs = parse_log(p)
        if not epochs:
            print(
                f"[!] Khong parse duoc dong nao tu {log_path} — kiem tra format file."
            )
            continue

        print(
            f"[OK] threshold={threshold}: {len(epochs)} epoch, "
            f"{len(test_epochs)} diem TEST, "
            f"loss cuoi={losses[-1]:.4f}"
            + (f", best Recall={max(recalls):.4f}" if recalls else "")
        )

        fig, (ax_loss, ax_metric) = plt.subplots(
            2,
            1,
            figsize=(7, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.08},
        )
        fig.patch.set_facecolor("white")
        ax_loss.set_facecolor("#FAFAFA")
        ax_metric.set_facecolor("#FAFAFA")

        plot_one(
            ax_loss, ax_metric, threshold, epochs, losses, test_epochs, recalls, ndcgs
        )

        fig.suptitle(
            f"LightGCN — ViFoodRec  (threshold={threshold})", fontsize=11, y=1.01
        )
        out_path = OUT_DIR / f"loss_curve_th{threshold.replace('.', '_')}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"    -> Da luu: {out_path}")

    # ---------- Bonus: 1 figure tong hop (4 subplots) ----------
    thresholds_found = [
        (th, Path(lp)) for th, lp in LOG_FILES.items() if Path(lp).exists()
    ]
    if len(thresholds_found) >= 2:
        fig, axes = plt.subplots(
            len(thresholds_found),
            2,
            figsize=(12, 3.5 * len(thresholds_found)),
            gridspec_kw={"hspace": 0.45, "wspace": 0.3},
        )
        fig.patch.set_facecolor("white")
        if len(thresholds_found) == 1:
            axes = [axes]

        for i, (th, lp) in enumerate(thresholds_found):
            epochs, losses, test_epochs, recalls, ndcgs = parse_log(lp)
            ax_loss, ax_metric = axes[i][0], axes[i][1]
            ax_loss.set_facecolor("#FAFAFA")
            ax_metric.set_facecolor("#FAFAFA")
            plot_one(
                ax_loss, ax_metric, th, epochs, losses, test_epochs, recalls, ndcgs
            )

        fig.suptitle(
            "LightGCN — ViFoodRec: So sanh 4 threshold binarize", fontsize=12, y=1.01
        )
        combined_path = OUT_DIR / "loss_curves_all_thresholds.png"
        fig.savefig(combined_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\n[+] Da luu hinh tong hop: {combined_path}")

    print("\nXong. Cac file PNG co the dua thang vao bao cao (Hinh 4.x).")


if __name__ == "__main__":
    main()
