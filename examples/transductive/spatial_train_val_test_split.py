
"""
任意のリモートセンシング画像に対し、空間ブロック分割による
train_val_test_split関数を実装する。(2025/11/27~)
"""

import numpy as np


def masks_to_indices(mask) -> np.ndarray:
    """
    True 位置の index (flatten) を返すユーティリティ関数
    """
    return np.where(mask.reshape(-1))[0]

# 進捗バー関数
def progress_bar(current: int, total: int, prefix: str = "", bar_length: int = 30):
    """
    current: 現在のループ数
    total: max_iter
    """
    fraction = current / total
    filled = int(fraction * bar_length)
    bar = "❚" * filled + "-" * (bar_length - filled)
    print(f"\r{prefix} |{bar}| {fraction*100:6.2f}%", end="")
    if current == total:
        print() # 最後だけ改行

class SpatialSplitConfig:
    def __init__(
        self,
        background_label = 0,
        min_train_samples_per_class = 2,
        min_val_samples_per_class = 2,
        min_test_samples_per_class = 2,
        max_iter_per_shape: int = 6000000,
        max_shapes: int = 100,
        min_block_height: int = 1,
        min_block_width: int = 1,
        min_total_blocks: int = 50,
        random_state: int | None = None,
        show_progress: bool = True,
    ):
        self.background_label = background_label
        self.min_train_samples_per_class = min_train_samples_per_class
        self.min_val_samples_per_class = min_val_samples_per_class
        self.min_test_samples_per_class = min_test_samples_per_class

        self.max_iter_per_shape = max_iter_per_shape # 一つの(N_rows, N_cols)に対する試行回数
        self.max_shapes = max_shapes                 # 何パターンの(N_rows, N_cols)を試すか
        self.min_block_height = min_block_height
        self.min_block_width = min_block_width
        self.min_total_blocks = min_total_blocks

        self.random_state = random_state
        self.show_progress = show_progress


def spatial_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.05,
    val_ratio: float = 0.05,
    test_ratio: float = 0.90,
    cfg : SpatialSplitConfig | None = None
):
    """
    train/val/test を空間ブロックによって同時に分割する。

    - 背景クラスは使用しない
    - train/val/test の各クラス最小サンプル数を満たす
    - n_rows/n_cols は block_candidates から選ぶ
    """
    if cfg is None:
        cfg = SpatialSplitConfig()

    if cfg.random_state is not None:
        np.random.seed(cfg.random_state)

    H, W = y.shape
    total_mask = (y != cfg.background_label)
    total_pixels = total_mask.sum()
    unique_classes = np.unique(y[total_mask])

    # Shape 候補の生成
    block_heights = [5, 10, 15, 20, 25, 30, 40, 50]
    block_widths  = [5, 10, 15, 20, 25, 30, 40, 50]

    pairs = []

    for bh in block_heights:
        if bh > H:
            continue
        N_rows = H // bh
        if N_rows < 2:
            continue

        for bw in block_widths:
            if bw > W:
                continue
            N_cols = W // bw
            if N_cols < 2:
                continue

            # ブロック総数
            total_blocks = N_rows * N_cols

            # 少ない shape は除外 (MIN_BLOCKS=50)
            if total_blocks < cfg.min_total_blocks:
                continue
            pairs.append((N_rows, N_cols, bh, bw, total_blocks))

    if len(pairs) == 0:
        raise ValueError("候補 shape が生成できませんでした")

    # 成功しやすい順にソート → ブロック数が多い順
    pairs.sort(key=lambda p: p[4], reverse = True)
    pairs = pairs[:cfg.max_shapes] # 探索数制限

    # 各 shape で探索
    for idx_shape, (N_rows, N_cols, bh, bw, total_blocks) in enumerate(pairs, start=1):

        print(f"\n[INFO] Shape {idx_shape}/{len(pairs)}: "
              f"N_rows={N_rows}, N_cols={N_cols}, blocks={total_blocks}")

        # ブロックリスト生成
        blocks = [(i, j) for i in range(N_rows) for j in range(N_cols)]
        num_blocks = len(blocks)

        num_train_blocks = max(1, int(num_blocks * train_ratio))
        num_val_blocks   = max(1, int(num_blocks * val_ratio))
        num_test_blocks  = num_blocks - num_train_blocks - num_val_blocks

        # shape 内探索 (早期終了版)
        for it in range(1, cfg.max_iter_per_shape + 1):
            if cfg.show_progress:
                progress_bar(it, cfg.max_iter_per_shape, prefix = "  試行")

            np.random.shuffle(blocks)

            train_blocks = set(blocks[:num_train_blocks])
            val_blocks   = set(blocks[num_train_blocks: num_train_blocks + num_val_blocks])
            test_blocks  = set(blocks[num_train_blocks + num_val_blocks:])

            # mask 準備
            train_mask = np.zeros((H, W), dtype = bool)
            val_mask   = np.zeros((H, W), dtype = bool)
            test_mask  = np.zeros((H, W), dtype = bool)

            # mask にブロックを割り当てる
            def assign(mask, s):
                for (r, c) in s:
                    mask[r*bh:(r+1)*bh, c*bw:(c+1)*bw] = True

            # def assign_mask(mask, block_set):
                # for (bi, bj) in block_set:
                    # mask[
                        # bi*block_h : (bi+1)*block_h,
                        # bj*block_w : (bj+1)*block_w
                    # ] = True

            assign(train_mask, train_blocks)
            assign(val_mask,   val_blocks)
            assign(test_mask,  test_blocks)

            # 背景除外
            train_mask &= total_mask
            val_mask   &= total_mask
            test_mask  &= total_mask

            # Step 3. クラスごとの最小サンプル数
            ok = True
            for cls in unique_classes:
                n_tr = np.sum(train_mask & (y == cls))
                n_va = np.sum(val_mask   & (y == cls))
                n_te = np.sum(test_mask  & (y == cls))

                if n_tr < cfg.min_train_samples_per_class:
                    ok = False; break
                if n_va < cfg.min_val_samples_per_class:
                    ok = False; break
                if n_te < cfg.min_test_samples_per_class:
                    ok = False; break

            if not ok:
                continue

            # Step 4. 比率の最適性評価
            ratio_tr = np.sum(train_mask) / total_pixels
            ratio_vr = np.sum(val_mask)   / total_pixels
            ratio_ts = np.sum(test_mask)  / total_pixels

            diff = abs(ratio_tr - train_ratio) \
                 + abs(ratio_vr - val_ratio) \
                 + abs(ratio_ts - test_ratio)

            # 条件良好 → 即終了
            if diff < 0.02: # 許容ズレ2%
                print("\n[INFO] → 成功したため早期終了 !")
                train_index = masks_to_indices(train_mask)
                val_index   = masks_to_indices(val_mask)
                test_index  = masks_to_indices(test_mask)

                return train_index, val_index, test_index, train_mask, val_mask, test_mask

        print("") # 進捗バーの改行

    # 全 shape で失敗
    raise ValueError("train/val/test の三分割に失敗しました。")
