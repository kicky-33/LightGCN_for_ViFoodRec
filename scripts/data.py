"""
data.py — Load dữ liệu dạng train.txt/test.txt chuẩn NGCF/LightGCN, dùng chung
cho BPR-MF / NGCF / LightGCN trên ViFoodRec (mọi threshold).

Xây normalized adjacency matrix theo công thức (6)-(7) trong paper LightGCN,
và sampler cho BPR training.
"""
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch


class InteractionData:
    """
    Format mỗi dòng trong train.txt / test.txt:
        user_id item_id1 item_id2 item_id3 ...
    token đầu tiên là user_id, các token còn lại là toàn bộ item user đó đã tương tác.
    """

    def __init__(self, train_path: str, test_path: str):
        self.train_user_items = defaultdict(list)
        self.test_user_items = defaultdict(list)

        self.n_users = 0
        self.n_items = 0
        self.n_train = 0
        self.n_test = 0

        self._read_file(train_path, self.train_user_items, is_train=True)
        self._read_file(test_path, self.test_user_items, is_train=False)

        # ids đếm từ 0 -> max id quan sát được + 1 = số lượng thực tế
        self.n_users += 1
        self.n_items += 1

        # danh sách user_id hợp lệ để sample (loại user không có item nào trong train)
        self.train_user_pool = [u for u, items in self.train_user_items.items() if len(items) > 0]

        # set hóa 1 lần để negative sampling tra cứu O(1) thay vì O(n) trên list
        self._train_items_set = {u: set(items) for u, items in self.train_user_items.items()}

        print(f"[InteractionData] n_users={self.n_users}, n_items={self.n_items}")
        print(f"[InteractionData] n_train_interactions={self.n_train}, n_test_interactions={self.n_test}")

    def _read_file(self, path: str, target_dict: dict, is_train: bool):
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) <= 1 or parts[0] == "":
                    continue
                u = int(parts[0])
                items = [int(i) for i in parts[1:] if i != ""]
                if len(items) == 0:
                    continue
                target_dict[u] = items
                self.n_users = max(self.n_users, u)
                self.n_items = max(self.n_items, max(items))
                if is_train:
                    self.n_train += len(items)
                else:
                    self.n_test += len(items)

    def build_norm_adj(self) -> torch.Tensor:
        """
        Xây A = [[0, R], [R^T, 0]] rồi chuẩn hóa A_tilde = D^(-1/2) A D^(-1/2)
        đúng theo công thức (6)-(7) trong paper LightGCN.
        Trả về sparse COO tensor.
        """
        R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, items in self.train_user_items.items():
            for i in items:
                R[u, i] = 1.0
        R = R.tocsr()

        zero_uu = sp.csr_matrix((self.n_users, self.n_users))
        zero_ii = sp.csr_matrix((self.n_items, self.n_items))
        A = sp.vstack(
            [sp.hstack([zero_uu, R]), sp.hstack([R.T, zero_ii])]
        ).tocsr()

        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.zeros_like(deg)
        nonzero_mask = deg > 0
        deg_inv_sqrt[nonzero_mask] = np.power(deg[nonzero_mask], -0.5)
        D_inv_sqrt = sp.diags(deg_inv_sqrt)

        A_norm = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

        indices = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
        values = torch.FloatTensor(A_norm.data)
        shape = torch.Size(A_norm.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def sample_batch(self, batch_size: int):
        """Sample (user, pos_item, neg_item) cho BPR loss."""
        users = np.random.choice(self.train_user_pool, batch_size)
        pos_items, neg_items = [], []
        for u in users:
            items = self.train_user_items[u]
            items_set = self._train_items_set[u]
            pos_items.append(random.choice(items))
            while True:
                neg = random.randint(0, self.n_items - 1)
                if neg not in items_set:
                    neg_items.append(neg)
                    break
        return (
            torch.LongTensor(users),
            torch.LongTensor(pos_items),
            torch.LongTensor(neg_items),
        )


# Alias để tương thích ngược nếu còn notebook/script cũ import tên YelpData
YelpData = InteractionData
