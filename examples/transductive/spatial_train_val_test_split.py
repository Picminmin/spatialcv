
"""
任意のリモートセンシング画像に対し、空間ブロック分割による
train_val_test_split関数を実装する。(2025/11/27~)
"""

import numpy as np


def masks_to_indices(mask):
    """
    True 位置の index (flatten) を返すユーティリティ関数
    """
    return np.where(mask.reshape(-1))[0]

class SpatialSplitConfig:
    def __init__(
        self,
        background_label = 0,
        min_train_samples_per_class = 3,
        min_val_samples_per_class = 2,
        min_test_samples_per_class = 3,
        max_iter = 200,
        block_candidates = None
    ):
        self.background_label = background_label
        self.min_train_samples_per_class = min_train_samples_per_class
        self.min_val_samples_per_class = min_val_samples_per_class
        self.min_test_samples_per_class = min_test_samples_per_class

        self.max_iter = max_iter

        # 分割ブロック候補 (大→小)
        if block_candidates is None:
            self.block_candidates = [13, 11, 9, 7, 5, 3, 2]
        else:
            self.block_candidates = block_candidates

def spatial_train_val_test_split(
    X, y,
    train_ratio = 0.05,
    val_ratio = 0.05,
    test_ratio = 0.90,
    cfg = SpatialSplitConfig()
):
    """
    train/val/test を空間ブロックによって同時に分割する。

    - 背景クラスは使用しない
    - train/val/test の各クラス最小サンプル数を満たす
    - n_rows/n_cols は block_candidates から選ぶ
    """
    H, W = y.shape
    unique_classes = np.unique(y[y != cfg.background_label])
    total_pixels = np.sum(y != cfg.background_label)

    best_split = None
    best_diff = float("inf")

    # Step 0. target_ratio に応じた推定 N_rows, N_cols
    target_ratio = min(train_ratio, val_ratio, test_ratio)

    # 最大ブロック数 (例: 13) から推定
    n_max = cfg.block_candidates[0]

    # √target_ratio を用いて推定
    est_N = max(2, int(np.round(n_max / np.sqrt(target_ratio))))
    est_N = min(min(H, W), est_N)

    # block_candidates を est_N を近い順に並べる
    # candidate_blocks_sorted = sorted(cfg.block_candidates, key=lambda x: abs(x - est_N), reverse = False)
    candidate_blocks_sorted = sorted(np.arange(est_N // 4, est_N, step = 1), reverse = False) # 降順
    # 例: est_N = 3 → candidate_blocks_sorted = [3, 2, 5, 7, 9, 11, 13]

    # Step 1. N_rows, N_cols 候補を順に試す
    for N in candidate_blocks_sorted:

        block_h = H // N
        block_w = W // N

        # 全ブロックのインデックス
        blocks = [(i, j) for i in range(N) for j in range(N)]
        num_blocks = len(blocks)

        # 必要ブロック数
        num_train_blocks = max(1, int(num_blocks * train_ratio))
        num_val_blocks = max(1, int(num_blocks * val_ratio))
        num_test_blocks = num_blocks - num_train_blocks - num_test_blocks

        if num_test_blocks <= 0:
            continue

        # Step 2. ランダム試行
        for _ in range(cfg.max_iter):

            np.random.shuffle(blocks)

            train_blocks = set(blocks[:num_train_blocks])
            val_blocks = set(blocks[num_train_blocks: num_train_blocks + num_val_blocks])
            test_blocks = set(blocks[num_train_blocks + num_val_blocks:])

            # mask 準備
            train_mask = np.zeros((H, W), dtype = bool)
            val_mask = np.zeros((H, W), dtype = bool)
            test_mask = np.zeros((H, W), dtype = bool)

            def assign_mask(mask, block_set):
                for (i, j) in block_set:
                    mask[
                        i*block_h : (i+1)*block_h,
                        j*block_w : (j+1)*block_w
                    ] = True

            assign_mask(train_mask, train_blocks)
            assign_mask(val_mask, val_blocks)
            assign_mask(test_mask, test_blocks)

            # 背景除外
            train_mask &= (y != cfg.background_label)
            val_mask   &= (y != cfg.background_label)
            test_mask  &= (y != cfg.background_label)

            # Step 3. クラスごとの最小サンプル数
            ok = True
            for cls in unique_classes:
                n_tr = np.sum(train_mask & (y == cls))
                n_va = np.sum(val_mask & (y == cls))
                n_te = np.sum(test_mask & (y == cls))

                if n_tr < cfg.min_train_samples_per_class:
                    ok = False; break
                if n_va < cfg.min_val_samples_per_class:
                    ok = False; break
                if n_te < cfg.min_test_samples_per_class:
                    ok = False; break

            if not ok:
                continue

            # Step 4. 比率の最適性評価
            tr = np.sum(train_mask) / total_pixels
            vr = np.sum(val_mask)   / total_pixels
            ts = np.sum(test_mask)  / total_pixels

            diff = abs(tr - train_ratio) + abs(vr - val_ratio) + abs(ts - test_ratio)

            if diff < best_diff:
                best_diff = diff
                best_split = (train_mask, val_mask, test_mask)

        if best_split is not None:
            train_mask, val_mask, test_mask = best_split
            tr, vr, ts = np.sum(train_mask) / total_pixels, \
                         np.sum(val_mask) / total_pixels, \
                         np.sum(test_mask) / total_pixels

            train_index = masks_to_indices(train_mask)
            val_index   = masks_to_indices(val_mask)
            test_index  = masks_to_indices(test_mask)
            print(f"[INFO]{100 * best_diff:.3f}%のずれ(全体)でtrain/val/test分割を行いました。")
            print("train/val/test誤差の内訳")
            print(f"[INFO]train: {100 * abs(tr - train_ratio):.3f}%の誤差")
            print(f"[INFO]val: {100 * abs(vr - val_ratio):.3f}%の誤差")
            print(f"[INFO]test: {100 * abs(ts - test_ratio):.3f}%の誤差")
            return train_index, val_index, test_index, train_mask, val_mask, test_mask

    raise ValueError("train/val/test の三分割に失敗しました。")



