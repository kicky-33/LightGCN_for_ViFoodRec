"""
Tien xu ly ViFoodRec (ratings.csv) thanh dinh dang train.txt / test.txt
cho LightGCN-PyTorch (https://github.com/gusye1234/LightGCN-PyTorch)

BAN NAY HO TRO CHAY NHIEU THRESHOLD TRONG 1 LAN
-> phuc vu thi nghiem "Anh huong cua nguong binarize" (xem plan_preprocessing.md)

Input: ratings.csv voi cot [userid, foodid, rating] (rating: 0.0 - 5.0, buoc 0.5)

Output, cho MOI threshold trong THRESHOLDS, tao 1 thu muc rieng
    OUTPUT_ROOT/th{threshold}/
        - train.txt, test.txt
        - userid_map.csv, foodid_map.csv
        - preprocessing_log.txt
Ngoai ra, tao them:
    OUTPUT_ROOT/summary_all_thresholds.csv
    -> bang tong hop #users/#items/#interactions/density cho CA 4 threshold,
       co the dua thang vao bao cao (Bang 4.3 - phan 4.3.2) MA KHONG CAN
       doi train xong moi co so lieu nay.

Quyet dinh phuong phap luan da thong nhat (xem note_han_che_du_lieu_ratings.md):
    - ratings.csv la ban da fill 40% gia tri thieu bang median (khong tach duoc) -> dung nguyen.
    - Dedupe user-item 1 lan duy nhat TRUOC khi binarize (rating khong doi theo threshold,
      lam 1 lan de tranh lap lai khong can thiet qua 4 threshold).
    - Dung CUNG 1 RANDOM_SEED cho train/test split o ca 4 threshold (control variable,
      dam bao khac biet ket qua la do threshold, khong phai do ngau nhien chia du lieu).

LUU Y VE QUY MO DU LIEU (quan trong khi doc ket qua):
    ViFoodRec ratings.csv chi co ~100 user (moi nguoi rate 436-566/4000 mon).
    -> K-core=5 GAN NHU KHONG LOC USER NAO o ca 4 threshold (kha nang con ~100 user
       o moi muc). Bien dong chinh nam o #items va #interactions, KHONG PHAI #users.
    Neu summary_all_thresholds.csv cho thay #users tut sau xa duoi ~100, hay kiem tra
    lai file ratings.csv dau vao truoc khi train (co the load nham file/sai cot).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ===================== CONFIG (chinh o day) =====================
RATINGS_PATH = "data/raw/ratings.csv"  # duong dan toi ratings.csv
OUTPUT_ROOT = "data/processed/experiments"  # thu muc goc chua ket qua 4 threshold
THRESHOLDS = [3.0, 3.5, 4.0, 4.5]  # nguong binarize can thu nghiem
K_CORE = 5  # nguong K-core (warm-start)
TEST_RATIO = 0.2  # ti le test moi user
RANDOM_SEED = 2026  # giu co dinh cho moi threshold (control variable)
# ==================================================================


def threshold_to_dirname(t):
    """3.0 -> 'th3_0', 4.5 -> 'th4_5' (khop voi cau truc thu muc experiments/ trong plan)"""
    return f"th{str(t).replace('.', '_')}"


def load_raw_dedup(path):
    """Doc ratings.csv 1 lan duy nhat, dedupe user-item (giu rating cao nhat neu trung)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"userid", "foodid", "rating"}.issubset(
        df.columns
    ), f"ratings.csv can co cot userid, foodid, rating. Hien co: {df.columns.tolist()}"

    n_before = len(df)
    df = df.sort_values("rating", ascending=False).drop_duplicates(
        subset=["userid", "foodid"], keep="first"
    )
    print(
        f"[Load] Tong dong ratings goc: {n_before} -> sau dedupe user-item: {len(df)}"
    )
    return df


def binarize(df, threshold):
    df_pos = df[df["rating"] >= threshold][["userid", "foodid"]].copy()
    print(
        f"  [Binarize >= {threshold}] {len(df_pos)} positive interactions "
        f"({len(df_pos) / len(df) * 100:.1f}% giu lai)"
    )
    return df_pos


def k_core_filter(df, k):
    """Lap loai bo user/item co so tuong tac < k cho toi khi on dinh."""
    prev_len = -1
    round_i = 0
    while prev_len != len(df) and len(df) > 0:
        prev_len = len(df)
        round_i += 1
        user_counts = df["userid"].value_counts()
        item_counts = df["foodid"].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        df = df[df["userid"].isin(keep_users) & df["foodid"].isin(keep_items)]
    print(
        f"  [K-core={k}] Hoi tu sau {round_i} vong lap -> con {len(df)} interactions, "
        f"{df['userid'].nunique()} users, {df['foodid'].nunique()} items"
    )
    return df


def remap_ids(df):
    userids = sorted(df["userid"].unique())
    item_ids = sorted(df["foodid"].unique())
    user2idx = {u: i for i, u in enumerate(userids)}
    item2idx = {f: i for i, f in enumerate(item_ids)}
    df = df.copy()
    df["u_idx"] = df["userid"].map(user2idx)
    df["i_idx"] = df["foodid"].map(item2idx)
    return df, user2idx, item2idx


def train_test_split_per_user(df, test_ratio, seed):
    """Split theo tung user (rng moi, cung seed cho moi threshold -> fair comparison)."""
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for u, group in df.groupby("u_idx"):
        items = group["i_idx"].tolist()
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio)) if len(items) >= 5 else 0
        test_items = items[:n_test]
        train_items = items[n_test:]
        if not train_items:
            train_items = test_items
            test_items = []
        train_rows.append((u, train_items))
        test_rows.append((u, test_items))
    return train_rows, test_rows


def write_lgn_format(rows, path):
    with open(path, "w") as f:
        for u, items in rows:
            if not items:
                continue
            f.write(" ".join([str(u)] + [str(i) for i in items]) + "\n")


def run_one_threshold(raw_dedup, threshold, out_dir, k_core, test_ratio, seed):
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pos = binarize(raw_dedup, threshold)
    df_core = k_core_filter(df_pos, k_core)

    if len(df_core) == 0:
        print(
            f"  [!] CANH BAO: threshold={threshold} khong con interaction nao sau K-core. Bo qua."
        )
        return {
            "threshold": threshold,
            "n_users": 0,
            "n_items": 0,
            "n_train": 0,
            "n_test": 0,
            "density": 0.0,
        }

    df_idx, user2idx, item2idx = remap_ids(df_core)
    train_rows, test_rows = train_test_split_per_user(df_idx, test_ratio, seed)

    write_lgn_format(train_rows, out_dir / "train.txt")
    write_lgn_format(test_rows, out_dir / "test.txt")

    pd.Series(user2idx).rename_axis("userid_goc").reset_index(name="useridx").to_csv(
        out_dir / "userid_map.csv", index=False
    )
    pd.Series(item2idx).rename_axis("foodid_goc").reset_index(name="foodidx").to_csv(
        out_dir / "foodid_map.csv", index=False
    )

    n_train = sum(len(items) for _, items in train_rows)
    n_test = sum(len(items) for _, items in test_rows)
    n_users, n_items = len(user2idx), len(item2idx)
    density = (n_train + n_test) / (n_users * n_items) if n_users and n_items else 0.0

    stats = {
        "threshold": threshold,
        "n_users": n_users,
        "n_items": n_items,
        "n_train": n_train,
        "n_test": n_test,
        "density": round(density, 6),
    }

    log_text = (
        f"=== THONG KE - threshold={threshold} ===\n"
        f"K-core: {k_core} | Test ratio: {test_ratio} | Seed: {seed}\n"
        f"So users: {n_users}\nSo items: {n_items}\n"
        f"Interactions train: {n_train}\nInteractions test: {n_test}\n"
        f"Density: {density:.6f}\n"
    )
    (out_dir / "preprocessing_log.txt").write_text(log_text)
    return stats


def main():
    raw_dedup = load_raw_dedup(RATINGS_PATH)

    summary_rows = []
    for t in THRESHOLDS:
        print(f"\n>>> Threshold = {t}")
        out_dir = Path(OUTPUT_ROOT) / threshold_to_dirname(t)
        stats = run_one_threshold(
            raw_dedup, t, out_dir, K_CORE, TEST_RATIO, RANDOM_SEED
        )
        summary_rows.append(stats)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(OUTPUT_ROOT) / "summary_all_thresholds.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("TONG HOP CA 4 THRESHOLD (dua thang vao Bang 4.3 trong bao cao):")
    print(summary_df.to_string(index=False))
    print("=" * 60)

    # Canh bao neu #users lech xa khoi ky vong (~100, xem docstring dau file)
    if (summary_df["n_users"] > 300).any():
        print(
            "\n[!] CANH BAO: co threshold ghi nhan #users > 300, lech xa so voi ky vong "
            "(~100 user trong ViFoodRec). Kiem tra lai file ratings.csv dau vao."
        )


if __name__ == "__main__":
    main()
