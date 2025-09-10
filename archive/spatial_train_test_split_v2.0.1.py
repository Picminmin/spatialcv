
import numpy as np
from sklearn.utils import resample

def spatial_train_test_split_v2(
    X,
    y,
    n_rows,
    n_cols,
    test_size = 0.3,
    min_acceptable_test_ratio = None,
    max_acceptable_test_ratio = None,
    background_label = 0,
    random_state = None,
    balance_classes = False,
    target_per_class = None,
    max_iter = 100
):
    """空間ブロック分割を用いたtrain/test分割。
    「各土地被覆クラスから少なくとも一つ以上の教師データを取得する」・・・①
    ①を満たしつつ、指定したtest_sizeに可能な限り近づける。

    ・spatial_train_test_splitに対する機能改善点
    1. 各土地被覆クラスに最低一つの教師データを必ず確保(最優先)
    2. 指定したtest_sizeにできるだけ近い割合で分割
    3. 許容範囲 min_acceptable_test_ratio / max_acceptable_test_ratio を設定可能
    4. 探索的にブロック割り当てを調整して最適な分割を見つける
    5. 実現不可能な場合は警告を出す

    Args:
        X (ndarray): 特徴量 (H, W, Bands)
        y (ndarray): ラベルマップ (H, W)
        n_row (int): ブロック分割数 (縦方向)
        n_cols (int): ブロック分割数 (横方向)
        test_size (float): テストデータの理想割合(0.0~1.0)
        min_acceptable_test_ratio (float): 許容されるテストデータの最小割合
        max_acceptable_test_ratio (float): 許容されるテストデータの最大割合
        background_label (int, optional): 背景ラベル
        random_state (_type_, optional): 乱数シード
        balance_classes (bool, optional): クラスバランスを揃えるかどうか
        target_per_class (_type_, optional): バランス調整時の上限サンプル数
        max_iter (int, optional): ブロック割り当てを再試行する最大回数
    """

    if random_state is not None:
        np.random.seed(random_state)

    # --- 初期設定 ---
    H, W = y.shape
    block_h = H // n_rows
    block_w = W // n_cols

    blocks = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    total_pixels = np.sum(y != background_label)

    # --- 許容範囲設定 ---
    if min_acceptable_test_ratio is None:
        min_acceptable_test_ratio = test_size
    if max_acceptable_test_ratio is None:
        max_acceptable_test_ratio = test_size

    best_diff = float("inf")
    best_result = None

    # --- 最大max_iter回まで試行 ---
    for _ in range(max_iter):
        np.random.shuffle(blocks)
        num_test_blocks = int(len(blocks) * test_size)
        test_blocks = set(blocks[:num_test_blocks])
        train_blocks = set(blocks[num_test_blocks:])

        # --- train/testマスク作成 ---
        train_mask = np.zeros_like(y, dtype = bool)
        test_mask = np.zeros_like(y, dtype = bool)

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

        # 特徴量とラベルを抽出
        X_flat, y_flat = X.reshape(-1, X.shape[2]), y.flatten()
        train_indices = np.where(train_mask.flatten())[0]
        test_indices = np.where(test_mask.flatten())[0]

        X_train, y_train = X_flat[train_indices], y_flat[train_indices]
        X_test, y_test = X_flat[test_indices], y_flat[test_indices]

        # --- 各土地被覆クラスの教師データが一つ以上あるか確認 ---
        unique_train = np.unique(y_train)
        unique_labels = np.unique(y[y != background_label])
        if not set(unique_labels).issubset(set(unique_train)):
            continue # 条件①を満たさないので再試行

        # --- test_sizeの実測値 ---
        actual_test_size = len(y_test) / total_pixels
        diff = abs(actual_test_size - test_size)

        # 許容範囲チェック
        if diff < best_diff and min_acceptable_test_ratio <= actual_test_size <= max_acceptable_test_ratio:
            best_diff = diff
            best_result = (
                X_train, X_test, y_train, y_test, train_mask, test_mask
            )

        if best_result is None:
            raise ValueError(
                f"条件を満たす分割が見つかりませんでした。 "
                f"test_size={test_size}, min={min_acceptable_test_ratio}, max={max_acceptable_test_ratio}を見直してください。"
            )

        # --- クラスバランスを揃えるオプション ---
        if balance_classes:
            X_train, y_train = _balance_classes(
                X_train, y_train, target_per_class, random_state
            )

        return best_result

    return X_train, y_train

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
