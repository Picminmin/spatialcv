
import numpy as np
from sklearn.utils import resample

def spatial_train_test_split(
    X,
    y,
    n_rows = 4,
    n_cols = 4,
    test_size = 0.2,
    background_label = 0,
    random_state = None,
    balance_classes = True,
    target_per_class = 200
):
    """空間ブロック分割を用いたtrain_test_split関数。
    オプションで「教師データの各土地被覆クラスのサンプル数をそろえる機能」を追加。

    Args:
        X (ndarray): 特徴量 (H, W, Bands)
        y (ndarray): ラベルマップ (H, W)
        n_rows (int): ブロック分割数（縦方向）
        n_cols (int): ブロック分割数（横方向）
        test_size (float): テストデータの割合 (0.0 ~ 1.0)
        background_label (int): 背景ラベル
        random_state (int): 乱数シード
        balance_classes (bool): クラスバランスを揃えるかどうか
        target_per_class (int): バランス調整時の上限サンプル数

    Returns:
        X_train, X_test, y_train, y_test, train_mask, test_mask
    """

    if random_state is not None:
        np.random.seed(random_state)

    H, W = y.shape
    block_h = H // n_rows
    block_w = W // n_cols

    # --- ブロック座標の取得 ---
    blocks = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    np.random.shuffle(blocks)

    # --- train/testブロックの分割 ---
    num_test_blocks = int(len(blocks) * test_size)
    test_blocks = set(blocks[:num_test_blocks])
    train_blocks = set(blocks[num_test_blocks:])

    # --- train/testマスク作成 ---
    train_mask = np.zeros_like(y, dtype = bool)
    test_mask = np.zeros_like(y, dtype = bool)

    for i,j in train_blocks:
        h_end = min((i+1)*block_h, H)
        w_end = min((j+1)*block_w, W)
        train_mask[i*block_h:h_end, j*block_w:w_end] = True

    for i,j in test_blocks:
        h_end = min((i+1)*block_h, H)
        w_end = min((j+1)*block_w, W)
        test_mask[i*block_h:h_end, j*block_w:w_end] = True

    # --- 背景を除外 ---
    train_mask = train_mask & (y != background_label)
    test_mask = test_mask & (y != background_label)

    # --- 特徴量とラベルを抽出 ---
    X_flat, y_flat = X.reshape(-1, X.shape[2]), y.flatten()
    train_indices = np.where(train_mask.flatten())[0]
    test_indices = np.where(test_mask.flatten())[0]

    X_train, y_train = X_flat[train_indices], y_flat[train_indices]
    X_test, y_test = X_flat[test_indices], y_flat[test_indices]

    # --- クラスバランスをそろえる処理 ---
    if balance_classes:
        mask = y_train != background_label
        unique_classes, counts = np.unique(y_train[mask], return_counts = True)
        print(f"[INFO] Before balancing: {dict(zip(unique_classes, counts))}")

        # ターゲットサンプル数の決定
        if target_per_class is None:
            target_per_class = np.min(counts)

        balanced_indices = []
        for cls in unique_classes:
            cls_indices = np.where(y_train == cls)[0]
            if len(cls_indices) > target_per_class:
                # サンプル数が多いクラスは間引く
                sampled = resample(
                    cls_indices,
                    n_samples = target_per_class,
                    replace = False,
                    random_state = random_state
                )
            else:
                # サンプル数が少ないクラスはそのまま
                sampled = cls_indices
            balanced_indices.extend(sampled)

        # バランス調整後のtrainデータ
        balanced_indices = np.array(balanced_indices)
        X_train = X_train[balanced_indices]
        y_train = y_train[balanced_indices]

        # 確認ログ
        mask = y_train != background_label
        unique_classes, counts = np.unique(y_train[mask], return_counts = True)
        print(f"[INFO] After balancing: {dict(zip(unique_classes, counts))}")

    return X_train, X_test, y_train, y_test, train_mask, test_mask
