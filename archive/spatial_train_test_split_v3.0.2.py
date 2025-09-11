
import numpy as np
from sklearn.utils import resample

def spatial_train_test_split_v3(
    X,
    y,
    n_rows,
    n_cols,
    test_size = 0.3,
    min_search_test_ratio = None,
    max_search_test_ratio = None,
    background_label = 0,
    random_state = None,
    balance_classes = False,
    target_per_class = None,
    auto_adjust_test_size = True,
    step = 0.05,
    max_iter = 100
):
    """"
    空間ブロック分割を用いたtrain/test分割。
    各土地被覆クラスに必ず一つ以上の教師データを含めつつ、
    指定したtest_sizeに可能な限り近づける。
    auto_adjust_test_size=Trueの場合、test_sizeを許容範囲内で自動探索。

    Args:
        X (ndarray): 特徴量 (H, W, Bands)
        y (ndarray): ラベルマップ (H, W)
        n_rows (int): ブロック分割数（縦方向）
        n_cols (int): ブロック分割数（横方向）
        test_size (float): テストデータの理想割合
        min_test_size (float): テストデータ割合の最小許容値
        max_test_size (float): テストデータ割合の最大許容値
        background_label (int): 背景ラベル
        random_state (int): 乱数シード
        balance_classes (bool): クラスバランスを揃えるかどうか
        target_per_class (int): バランス調整時の上限サンプル数
        auto_adjust_test_size (bool): test_sizeが不適切な場合に許容範囲で探索するか
        step (float): test_size探索時の刻み幅
        max_iter (int): ランダム分割の最大試行回数

    Returns:
        X_train, X_test, y_train, y_test, train_mask, test_mask, best_test_size
    """
    if random_state is not None:
        np.random.seed(random_state)

    H, W = y.shape
    block_h = H // n_rows
    block_w = W // n_cols
    blocks = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    total_pixels = np.sum(y != background_label)

    # 背景を除いた全クラス
    all_classes = np.unique(y[y != background_label])

    # 許容範囲設定
    if min_search_test_ratio is None:
        min_search_test_ratio = test_size
    if max_search_test_ratio is None:
        max_search_test_ratio = test_size

    # --- 内部関数：1回の試行で分割を作成 ---
    def try_split(ts):
        np.random.shuffle(blocks)
        num_test_blocks = int(len(blocks) * ts)
        test_blocks = set(blocks[:num_test_blocks])
        train_blocks = set(blocks[num_test_blocks:])

        # train/testマスク生成
        train_mask = np.zeros_like(y, dtype=bool)
        test_mask = np.zeros_like(y, dtype=bool)
        for i, j in train_blocks:
            h_end = min((i+1)*block_h, H)
            w_end = min((j+1)*block_w, W)
            train_mask[i*block_h:h_end, j*block_w:w_end] = True
        for i, j in test_blocks:
            h_end = min((i+1)*block_h, H)
            w_end = min((j+1)*block_w, W)
            test_mask[i*block_h:h_end, j*block_w:w_end] = True

        # 背景を除外
        train_mask &= (y != background_label)
        test_mask &= (y != background_label)

        # インデックス取得
        X_flat = X.reshape(-1, X.shape[2])
        y_flat = y.flatten()
        train_indices = np.where(train_mask.flatten())[0]
        test_indices = np.where(test_mask.flatten())[0]
        X_train = X_flat[train_indices]
        y_train = y_flat[train_indices]
        X_test = X_flat[test_indices]
        y_test = y_flat[test_indices]

        # trainに全クラスが含まれるかチェック
        train_classes = np.unique(y_train)
        if not set(all_classes).issubset(set(train_classes)):
            return None

        return X_train, X_test, y_train, y_test, train_mask, test_mask

    # --- 探索開始 ---
    candidate_sizes = (
        np.arange(min_search_test_ratio, max_search_test_ratio + 1e-8, step)
        if auto_adjust_test_size else [test_size]
    )

    best_result = None
    best_size = None
    best_diff = float("inf")

    for ts in candidate_sizes:
        for _ in range(max_iter):
            result = try_split(ts)
            if result is None:
                continue

            # 実際のtest_sizeとの誤差を評価
            X_train, X_test, y_train, y_test, train_mask, test_mask = result
            actual_ts = len(y_test) / total_pixels
            diff = abs(actual_ts - test_size)

            if diff < best_diff:
                best_result = result
                best_size = actual_ts
                best_diff = diff
                if diff < 1e-3:
                    break  # 理想値に十分近いなら打ち切り

        if best_result is not None:
            break  # 許容範囲内で見つかったら終了

    if best_result is None:
        raise ValueError(
            f"条件を満たす分割が見つかりませんでした。 "
            f"test_size={test_size}, min={min_search_test_ratio}, max={max_search_test_ratio} を見直してください。"
        )

    # --- クラスバランスを揃えるオプション ---
    if balance_classes:
        X_train, y_train = _balance_classes(
            best_result[0], best_result[2], target_per_class, random_state
        )
        best_result = (X_train, best_result[1], y_train, best_result[3], best_result[4], best_result[5])

    print(f"[INFO] 最適な test_size ≈ {best_size:.3f}")
    return (*best_result, best_size)


def _balance_classes(X_train, y_train, target_per_class, random_state):
    """クラスバランス調整用のヘルパー関数"""
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"[INFO] Before balancing: {dict(zip(unique_classes, counts))}")

    if target_per_class is None:
        target_per_class = np.min(counts)

    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        if len(cls_indices) > target_per_class:
            sampled = resample(
                cls_indices,
                n_samples=target_per_class,
                replace=False,
                random_state=random_state
            )
        else:
            sampled = cls_indices
        balanced_indices.extend(sampled)

    balanced_indices = np.array(balanced_indices)
    X_train = X_train[balanced_indices]
    y_train = y_train[balanced_indices]

    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"[INFO] After balancing: {dict(zip(unique_classes, counts))}")

    return X_train, y_train
