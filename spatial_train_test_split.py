
"""
任意のリモートセンシング画像に対し、空間ブロック分割による
train_test_split関数を実装する。(2025/08/29~)

backgroundを含むリモートセンシング画像を使った土地被覆分類手法の開発では、
backgroundはtrain/testのいずれにも割り当てないようにしなければならない。

そのため、次の条件を満たすような空間ブロック分割によるtrain_test_split関数の
実装が求められる。

1) backgroundを除外した有効画素だけを対象にtrain_test_splitをする
   (nobgは'no background'の略で、背景を除外したデータを意味する)
2) さらに、train/testのサンプル数は任意のtest_size(テストデータの割合)に近くなるようにする
3) また、train/testはブロック単位で空間的に分離したい

・spatial_train_test_splitの実装方針
1. まずbackground込みでブロック分割する
2. 各ブロック内の有効画素数(backgroundでない要素)を数える
3. ブロック単位でtrain/testを決定
4. train/testに割り当てた後、background画素は除外

"""

import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from matplotlib.colors import ListedColormap, to_rgba

__version__ = "0.0.1"

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

def spatial_train_test_split_v4(
    X,
    y,
    n_rows,
    n_cols,
    test_size = 0.3,
    min_train_samples_per_class = 1,
    min_test_samples_per_class = 5,
    background_label = 0,
    random_state = None,
    auto_adjust_test_size = False,
    min_search_test_ratio = None,
    max_search_test_ratio = None,
    step = 0.05,
    max_iter = 100
):
    """
    空間ブロック分割によるtrain/test分割(v4)
    spatial_train_test_split_v4は以下の①、②を満たす。
    「各土地被覆クラスで最低限の教師データのサンプル数を保証する。」・・・①
    「各土地被覆クラスで最低限のテストサンプル数を保証する。」・・・②
    ②より、CA(Class Accuracy)が0となる土地被覆クラスではCA自体は未定義ではなく、
    分類器の設計による誤分類が起こっていることが確かめられるようになる。
    分類器が誤分類を起こすのを減らすような土地被覆分類手法を考察するにあたり、実装が必要
    な部分である。
    v3までの空間ブロック分割によるtrain/test分割ではtestデータのサンプルが
    存在しないような土地被覆クラスがあるのを許していた、ともいえる。

    Args:
        X (ndarray): 特徴量 (H, W, Bands)
        y (ndarray): ラベルマップ (H, W)
        n_rows (int): ブロック分割数(縦)
        n_cols (int): ブロック分割数(横)
        test_size (float, optional): 理想的なテストデータの割合(0.0~1.0)
        min_train_samples_per_class (int, optional): 各土地被覆クラスで最低限必要な教師データのサンプル数
        min_test_samples_per_class (int, optional): 各土地被覆クラスで最低限必要なテストサンプル数
        background_label (int, optional): 背景クラスラベル
        random_state (int): 乱数シード
        auto_adjust_test_size (bool, optional): テストサイズを自動調整するか
        min_search_test_ratio (float, optional): 自動探索モード時の下限
        max_search_test_ratio (float, optional): 自動探索モード時の上限
        step (float, optional): 自動探索モード時の刻み幅
        max_iter (int, optional): ランダム試行回数

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
    unique_classes = np.unique(y[y != background_label])

    def try_split(ts):
        """指定されたtest_sizeで1回分割を試行"""
        np.random.shuffle(blocks)
        num_test_blocks = int(len(blocks) * ts)
        test_blocks = set(blocks[:num_test_blocks])
        train_blocks = set(blocks[num_test_blocks:])

        train_mask = np.zeros_like(y, dtype = bool)
        test_mask = np.zeros_like(y, dtype = bool)

        for i, j in train_blocks:
            h_end = min((i + 1) * block_h, H)
            w_end = min((j + 1) * block_w, W)
            train_mask[i * block_h:h_end, j * block_w:w_end] = True

        for i, j in test_blocks:
            h_end = min((i + 1) * block_h, H)
            w_end = min((j + 1) * block_w, W)
            test_mask[i * block_h:h_end, j * block_w:w_end] = True

        # 背景を除去
        train_mask &= (y != background_label)
        test_mask &= (y != background_label)

        # flatten
        X_flat = X.reshape(-1, X.shape[2])
        y_flat = y.flatten()

        train_indices = np.where(train_mask.flatten())[0]
        test_indices = np.where(test_mask.flatten())[0]

        X_train, y_train = X_flat[train_indices], y_flat[train_indices]
        X_test, y_test = X_flat[test_indices], y_flat[test_indices]

        # 各クラスのテストサンプル数をチェック
        for cls in unique_classes:
            cls_train_count = np.sum(y_train == cls)
            cls_test_count = np.sum(y_test == cls)
            if cls_train_count < min_train_samples_per_class:
                return None # 条件を満たさない → 再試行
            if cls_test_count < min_test_samples_per_class:
                return None # 条件を満たさない → 再試行

        return X_train, X_test, y_train, y_test, train_mask, test_mask

    # =========================================
    # auto_adjust_test_size = Falseの場合(固定)
    # =========================================

    if not auto_adjust_test_size:
        for _ in range(max_iter):
            result = try_split(test_size)
            if result is not None:
                best_test_size = len(result[1]) / total_pixels
                return (*result, best_test_size)

        raise ValueError(
            f"条件を満たす分割が見つかりませんでした。"
            f"test_size={test_size}, min_test_samples_per_class={min_test_samples_per_class}"
        )
    # =========================================
    # auto_adjust_test_size = Trueの場合(探索)
    # =========================================
    min_ratio = min_search_test_ratio or test_size
    max_ratio = max_search_test_ratio or test_size
    candidate_sizes = np.arange(min_ratio, max_ratio + 1e-8, step)

    best_result = None
    best_test_size = None
    best_diff = float("inf")

    for ts in candidate_sizes:
        for _ in range(max_iter):
            result = try_split(ts)
            if result is None:
                continue
            X_train, X_test, y_train, y_test, train_mask, test_mask = result
            actual_ts = len(y_test) / total_pixels
            diff = abs(actual_ts - test_size)
            if diff < best_diff:
                best_result = result
                best_test_size = actual_ts
                best_diff = diff
            if diff < 1e-3:
                break

    if best_result is None:
        raise ValueError(
            f"条件を満たす分割が見つかりませんでした。探索範囲を広げてください。"
        )

    return (*best_result, best_test_size)



def visualize_train_test_blocks(
    y,
    train_mask,
    test_mask,
    background_label = 0,
    title = "Spatial Trian/Test Split Visualization"
):
    """
    教師データ・テストデータの空間配置をオーバーレイ表示する関数

    <動作状況>
    教師データ・テストデータの空間配置を可視化できる。
    Args:
        y (ndarray): ラベルマップ (H, W)
        train_mask (ndarray): 教師データのマスク (H, W)
        test_mask (ndarray): テストデータのマスク (H, W)
        background_label (int): 背景クラスラベル
        title (str, optional): グラフタイトル
    """

    # 可視化用配列作成: 背景はnan, train = 1, test = 2
    display = np.full(y.shape, np.nan, dtype = float)
    display[train_mask] = 1
    display[test_mask] = 2

    # カラーマップを定義(train = 青, test = 赤)
    cmap = ListedColormap(["#1f77b4","#d62728"])

    # ブロット設定
    plt.figure(figsize = (7,7))
    im = plt.imshow(display, cmap = cmap, interpolation = "none")
    plt.title(
        f"{title}\nTrain={np.sum(train_mask)} / Test={np.sum(test_mask)}",
        fontsize = 12
    )
    plt.axis("off")

    # 凡例設定
    cbar = plt.colorbar(im, ticks = [1,2])
    cbar.ax.set_yticklabels(["Train","Test"])
    plt.show()

def visualize_train_test_with_counts(
    y,
    train_mask,
    test_mask,
    background_label = 0,
    class_names = None,
    title = 'Train/Test Spatial Split Visualization',
    ground_alpha = 0.4,
    train_alpha = 1.0
):

    """
    教師データ・テストデータの空間配置をオーバーレイ表示し、
    クラスごとの教師データ数・テストデータ数を棒グラフで表示する。
    Args:
        y (ndarray): ラベルマップ (H, W)
        train_mask (ndarray): 教師データのマスク (H, W)
        test_mask (ndarray): テストデータのマスク (H, W)
        background_label (int, optional): 背景クラスラベル
        class_names (_type_, optional): クラス名リスト(Noneの場合はクラスIDを表示)
        title (str, optional): グラフタイトル
        ground_alpha (float, optional): ground truthの領域の色の濃さ
        train_alpha (float, optional): trainデータの領域の色の濃さ
    """

    unique_classes = np.unique(y[y != background_label])
    n_classes = len(unique_classes)

    # クラスごとの色を生成
    cmap = plt.cm.get_cmap("tab20", n_classes)
    base_colors = [cmap(i) for i in range(n_classes)]

    # ====================================
    # (1) オーバーレイ図
    # ====================================

    # 表示用画像(背景は透明)
    display = np.full((*y.shape, 4),np.nan) # RGBA画像
    for idx, cls in enumerate(unique_classes):
        cls_mask = (y == cls)
        # ground truth (ベースカラー)
        display[cls_mask] = to_rgba(base_colors[idx], alpha = ground_alpha)
        # 教師データはground truthより濃くする
        # np.logical_and()で「あるクラスclsに属し、かつ教師データに割り当てられているピクセル」をTrueにする
        display[np.logical_and(cls_mask, train_mask)] = to_rgba(base_colors[idx], alpha = train_alpha)

    # 可視化
    plt.figure(figsize = (9, 9))
    plt.imshow(display)
    plt.title(f"{title}\nTrain={np.sum(train_mask)}, Test={np.sum(test_mask)}",fontsize = 12)
    plt.axis("off")

    # 凡例作成
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=base_colors[i], markersize = 10)
        for i in range(n_classes)
    ]
    # class_names[i] → class_names[i+1] (class_namesの0番目にbackgroundのあるデータはこのように対処する)
    labels = [class_names[i+1] if class_names is not None else f"Class {cls}"
              for i, cls in enumerate(unique_classes)]
    plt.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad = 0.,
        fontsize = 9
    )
    overlay_path = "./img/spatial_split_overlay.png"
    plt.savefig(overlay_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ====================================
    # (2) 棒グラフ
    # ====================================
    train_counts = [np.sum(np.logical_and(y == cls, train_mask)) for cls in unique_classes]
    test_counts = [np.sum(np.logical_and(y == cls, test_mask)) for cls in unique_classes]

    x = np.arange(n_classes)
    bar_width = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - bar_width/2, train_counts, bar_width, label="Train", color="#1f77b4")
    plt.bar(x + bar_width/2, test_counts, bar_width, label="Test", color="#d62728")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Sample Count")
    plt.title("Class-wise Train/Test Distribution")
    plt.legend()

    # 棒グラフ上に値を表示
    for i, v in enumerate(train_counts):
        plt.text(i - bar_width/2, v + 1, str(v), ha="center", fontsize = 9)
    for i, v in enumerate(test_counts):
        plt.text(i + bar_width/2, v + 1, str(v), ha="center", fontsize = 9)
    bargraph_path = "./img/class_distribution.png"
    plt.tight_layout()
    plt.savefig(bargraph_path, dpi=300)
    plt.close()

    print(f"[INFO] オーバーレイ図を保存しました: {overlay_path}")
    print(f"[INFO] 棒グラフを保存しました: {bargraph_path}")


def visualize_train_test_with_counts_csv(
    y,
    train_mask,
    test_mask,
    background_label=0,
    csv_path=None,
    save_dir="img",
    title="Train/Test Spatial Split Visualization"
):
    """
    教師データ・テストデータの空間オーバーレイ図とクラス分布棒グラフを作成する。
    CSVファイルでカラーマップを定義可能。

    Args:
        y (ndarray): ラベルマップ (H, W)
        train_mask (ndarray): 教師データのマスク (H, W)
        test_mask (ndarray): テストデータのマスク (H, W)
        background_label (int): 背景クラスラベル
        csv_path (str): カラーマップ定義CSVのパス
        save_dir (str): 結果画像の保存先
        title (str): 図のタイトル
    """
    os.makedirs(save_dir, exist_ok=True)

    unique_classes = np.unique(y[y != background_label])
    n_classes = len(unique_classes)

    # ==============================
    # (1) カラーマップを読み込み
    # ==============================
    if csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        color_dict = dict(zip(df["class_id"], df["color"]))
        name_dict = dict(zip(df["class_id"], df["name"]))
        base_colors = [to_rgba(color_dict.get(cls, "#808080")) for cls in unique_classes]
        labels = [name_dict.get(cls, f"Class {cls}") for cls in unique_classes]
    else:
        print("[WARN] CSVファイルが見つからないため、デフォルトのcmapを使用します。")
        cmap = plt.cm.get_cmap("tab20", n_classes)
        base_colors = [cmap(i) for i in range(n_classes)]
        labels = [f"Class {cls}" for cls in unique_classes]

    # ==============================
    # (2) 空間オーバーレイ図
    # ==============================
    display = np.full((*y.shape, 4), np.nan)
    for idx, cls in enumerate(unique_classes):
        cls_mask = (y == cls)
        display[cls_mask] = to_rgba(base_colors[idx], alpha=0.6)
        display[np.logical_and(cls_mask, train_mask)] = to_rgba(base_colors[idx], alpha=1.0)

    plt.figure(figsize=(8, 8))
    plt.imshow(display)
    plt.title(f"{title}\nTrain={np.sum(train_mask)}, Test={np.sum(test_mask)}", fontsize=12)
    plt.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=base_colors[i], markersize=10)
        for i in range(n_classes)
    ]
    plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)

    overlay_path = os.path.join(save_dir, "spatial_split_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=300)
    plt.close()

    # ==============================
    # (3) クラスごとのサンプル数棒グラフ
    # ==============================
    train_counts = [np.sum(np.logical_and(y == cls, train_mask)) for cls in unique_classes]
    test_counts = [np.sum(np.logical_and(y == cls, test_mask)) for cls in unique_classes]

    x = np.arange(n_classes)
    bar_width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width / 2, train_counts, bar_width, label="Train", color="#1f77b4")
    plt.bar(x + bar_width / 2, test_counts, bar_width, label="Test", color="#d62728")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Sample Count")
    plt.title("Class-wise Train/Test Distribution")
    plt.legend()

    # 棒グラフ上にサンプル数を表示
    for i, v in enumerate(train_counts):
        plt.text(i - bar_width/2, v + 1, str(v), ha="center", fontsize=9)
    for i, v in enumerate(test_counts):
        plt.text(i + bar_width/2, v + 1, str(v), ha="center", fontsize=9)

    bargraph_path = os.path.join(save_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(bargraph_path, dpi=300)
    plt.close()

    print(f"[INFO] オーバーレイ図を保存しました: {overlay_path}")
    print(f"[INFO] クラス分布棒グラフを保存しました: {bargraph_path}")


def load_mat_file(file_path):
    """MATファイルの読み込み"""
    try:
        # MATファイルを読み込む
        mat_data = scipy.io.loadmat(file_path)

        # 不要なメタデータを除外(MATLAB形式のファイルにはメタデータが含まれることがある)
        mat_data_cleaned = {key: value for key, value in mat_data.items() if not
                            key.startswith('__')}
        return mat_data_cleaned
    except FileNotFoundError:
        print(f'エラー: ファイルが見つかりません:{file_path}')
    except Exception as e:
        print(f'エラーが発生しました:{e}')

if __name__ == '__main__':
    from pprint import pprint
    import scipy.io
    from _Pixel import PixelDataset
    from _data_utils import get_default_pines

    """
    ・問題提起
    'Hyperspectral Remote Sensing Scenes'
    URL: https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
    で公開されているリモートセンシングデータに関して、「gt(ground truth)とcorrectedの違いは何か。」

    ナイジェリアのLULC解析に関する査読論文のsystematic reviewがなされた
    'Land use land cover(LULC) analysis in Nigeria: a systematic review of data,
    methods, and platforms with future prospects'
    には、gt(ground truth)はなく、'pre-classified LULC data'という用語が使われている。

    ChatGPTにそれぞれの用語の相違について尋ねると、以下の回答が得られた。
    gt(ground truth):「現在の地表状態をもっとも正確に表したデータ」
    'pre-classified LULC data': 「ある時点、あるプロジェクトの基準に基づいて、各画素に土地被覆
    クラスを割り当てたデータ」

    以上を踏まえると、correctedは'pre-classified LULC data'に相当する可能性があると私は思う。
    """

    def show_rs_info(features, gt=None, name='Dataset'):
        """リモートセンシング画像に対する空間ブロック分割の動作確認に関する情報を表示する

        Args:
            features (ndarray): スペクトル特徴量 (H,W,B)

            gt (ndarray,optional): Ground Truth (H,W)
            name (str): データセット名
        """
        features, gt = features, gt
        background_label = 0
        # (1)spatial_train_test_splitでの動作確認
        # X_train, X_test, y_train, y_test, train_mask, test_mask = spatial_train_test_split(
            # X = features,
            # y = gt,
            # n_rows = 2,
            # n_cols = 3,
            # test_size = 0.7,
            # background_label = 0,
            # balance_classes= True,
            # target_per_class = 100,
            # random_state = 2
        # )

        # (2)spatial_train_test_split_v2での動作確認
        # test_size = 0.7
        # X_train, X_test, y_train, y_test, train_mask, test_mask = spatial_train_test_split_v2(
            # X = features,
            # y = gt,
            # n_rows = 13,
            # n_cols = 13,
            # test_size = test_size,
            # background_label = 0,
            # min_acceptable_test_ratio = test_size - 0.5,
            # max_acceptable_test_ratio = test_size + 0.5,
            # random_state = 2,
            # balance_classes = True,
            # target_per_class = 100,
            # max_iter = 100
        # )

        # (3)spatial_train_test_split_v3での動作確認
        # test_size, error_rate = 0.7, 0.05
        # X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split_v3(
            # X = features,
            # y = gt,
            # n_rows = 13,
            # n_cols = 13,
            # test_size = test_size,
            # min_search_test_ratio = test_size - error_rate,
            # max_search_test_ratio = test_size + error_rate,
            # background_label = 0,
            # random_state = None,
            # balance_classes = False,
            # target_per_class = None,
            # auto_adjust_test_size = True,
            # step = 0.05,
            # max_iter = 100
        # )

        # (4)spatial_train_test_split_v4での動作確認
        test_size, error_rate = 0.7, 0.1
        X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split_v4(
            X = features,
            y = gt,
            n_rows = 13,
            n_cols = 13,
            test_size = test_size,
            min_search_test_ratio = test_size - error_rate,
            max_search_test_ratio = test_size + error_rate,
            background_label = 0,
            min_test_samples_per_class = 10, # 各クラスで最低10サンプルのテストデータを保証
            auto_adjust_test_size = True,
            random_state = 42,
            step = 0.05,
            max_iter = 100
        )

        nobg_count = gt[gt != background_label].size
        split_name = 'spatial_train_test_split_v3'
        print(f"----- {split_name}での動作確認 -----")
        print(f'{name}データセット全体のサンプル数:{gt.flatten().size}')
        print(f'{name}特徴量の形状: features: {features.shape}, gt: {gt.shape}')
        print(f'backgroundを除外した{name}のサンプル数:{nobg_count}')
        print(f'spatial_train_test_split適用後のサンプル数の内訳と総数')
        print(f"内訳(tr:{X_train.shape[0]},te:{X_test.shape[0]})',総数:{X_train.shape[0] + X_test.shape[0]}")
        print(f'test_size by sample_num:{int(X_test.shape[0]) / nobg_count:.3f}')

        return gt, train_mask, test_mask

    # --- 利用可能なリモートセンシング画像に対する空間ブロック分割のログ出力 ---
    ## (1)Indianpinesデータに対する空間ブロック分割

    pines, train_test_status, train_index, test_index, background = get_default_pines()
    dataset = PixelDataset(pines_data = pines, train_test_status = train_test_status)
    pines_h,pines_w,pines_f = 145, 145, 200
    name = 'Indianpines'
    X = pines.features.reshape([pines_h, pines_w, pines_f])
    y = pines.target.reshape([X.shape[0], X.shape[1]])
    y, train_mask, test_mask = show_rs_info(features=X, gt=y, name=name)

    # visualize_train_test_with_counts_csv(
        # y = y,
        # train_mask = train_mask,
        # test_mask = test_mask,
        # background_label=0,
        # csv_path = r'C:\Users\sohta\Documents\大学院研究\MasterProgram_from20250724\RS_GroundTruth\01_Indian Pines\indianpines_class_colors.csv',
        # title = name
    # )

    # (2)Paviaに対する空間ブロック分割
    # name = 'Pavia_data'
    # Pavia_data = load_mat_file('./RS_GroundTruth/03_Pavia Centre and University/Pavia.mat')
    # Pavia_data_gt = load_mat_file('./RS_GroundTruth/03_Pavia Centre and University/Pavia_gt.mat')
    # pprint(f'Pavia_data.keys():{Pavia_data.keys()}')
    # pprint(f'Pavia_data_gt.keys():{Pavia_data_gt.keys()}')
    # pavia_features = Pavia_data['pavia'] # (610, 340, 103)
    # pavia_target = Pavia_data_gt['pavia_gt'] # (610, 340)
    # y, train_mask, test_mask = show_rs_info(
    # features = pavia_features,
    # gt = pavia_target,
    # name = name
    # )
    # visualize_train_test_with_counts_csv(
        # y = y,
        # train_mask = train_mask,
        # test_mask = test_mask,
        # background_label=0,
        # csv_path = r'C:\Users\sohta\Documents\大学院研究\MasterProgram_from20250724\RS_GroundTruth\03_Pavia Centre and University\pavia_class_colors.csv',
        # title = name
    # )

    # (3) PaviaUに対する空間ブロック分割

    # name = 'PaviaU_data'
    # PaviaU_data = load_mat_file('./RS_GroundTruth/03_Pavia Centre and University/PaviaU.mat')
    # PaviaU_data_gt = load_mat_file('./RS_GroundTruth/03_Pavia Centre and University/PaviaU_gt.mat')
    # pprint(f'PaviaU_data.keys():{PaviaU_data.keys()}')
    # pprint(f'PaviaU_data_gt.keys():{PaviaU_data_gt.keys()}')
    # pavia_features = PaviaU_data['paviaU'] # (610, 340, 103)
    # pavia_target = PaviaU_data_gt['paviaU_gt'] # (610, 340)
    # y, train_mask, test_mask = show_rs_info(
    # features = pavia_features,
    # gt = pavia_target,
    # name = name
    # )

    # visualize_train_test_with_counts_csv(
        # y = y,
        # train_mask = train_mask,
        # test_mask = test_mask,
        # background_label=0,
        # csv_path = r'C:\Users\sohta\Documents\大学院研究\MasterProgram_from20250724\RS_GroundTruth\03_Pavia Centre and University\paviaU_class_colors.csv',
        # title = name
    # )

    # (4) Salinasに対する空間ブロック分割

    # name = 'Salinas_data'
    # Salinas_data = load_mat_file('./RS_GroundTruth/02_Salinas/Salinas.mat')
    # Salinas_data_gt = load_mat_file('./RS_GroundTruth/02_Salinas/Salinas_gt.mat')
    # pprint(f'Salinas_data.keys():{Salinas_data.keys()}')
    # pprint(f'Salinas_data_gt.keys():{Salinas_data_gt.keys()}')
    # Salinas_features = Salinas_data['salinas'] # (512, 217, 224)
    # Salinas_target = Salinas_data_gt['salinas_gt'] # (512, 217)
    # y, train_mask, test_mask = show_rs_info(
    # features = Salinas_features,
    # gt = Salinas_target,
    # name = name
    # )

    # visualize_train_test_with_counts_csv(
        # y = y,
        # train_mask = train_mask,
        # test_mask = test_mask,
        # background_label=0,
        # csv_path = r'C:\Users\sohta\Documents\大学院研究\MasterProgram_from20250724\RS_GroundTruth\02_Salinas\salinas_class_colors.csv',
        # title = name
    # )

    # (5) SalinasAに対する空間ブロック分割
    name = 'SalinasA_data'
    SalinasA_data = load_mat_file('./RS_GroundTruth/02_Salinas/SalinasA.mat')
    SalinasA_data_gt = load_mat_file('./RS_GroundTruth/02_Salinas/SalinasA_gt.mat')
    pprint(f'SalinasA_data.keys():{SalinasA_data.keys()}')
    pprint(f'SalinasA_data_gt.keys():{SalinasA_data_gt.keys()}')
    SalinasA_features = SalinasA_data['salinasA'] # (83, 86, 224)
    SalinasA_target = SalinasA_data_gt['salinasA_gt'] # (83, 86)
    y, train_mask, test_mask = show_rs_info(
    features = SalinasA_features,
    gt = SalinasA_target,
    name = name
    )

    visualize_train_test_with_counts_csv(
        y = y,
        train_mask = train_mask,
        test_mask = test_mask,
        background_label=0,
        csv_path = r"C:\Users\sohta\Documents\大学院研究\MasterProgram_from20250724\RS_GroundTruth\02_Salinas\salinasA_class_colors.csv",
        title = name
    )
