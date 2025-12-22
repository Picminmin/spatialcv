
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
from dataclasses import dataclass
from matplotlib.colors import ListedColormap, to_rgba

__version__ = "0.0.5"


@dataclass
class SpatialSplitConfig:
    n_rows = 13                       # ブロック分割数(縦)
    n_cols = 13                       # ブロック分割数(横)
    min_train_samples_per_class = 20  # 各土地被覆クラスで作成する教師データのサンプル数の下限
    min_test_samples_per_class = 5    # 各土地被覆クラスで作成するなテストサンプル数の下限
    background_label = 0              # 各土地被覆クラスで作成するなテストサンプル数の下限
    random_state = None               # 乱数シード
    auto_adjust_test_size = False     # テストサイズを自動調整するか
    min_search_test_ratio = None      # 自動探索モード時の下限
    max_search_test_ratio = None      # 自動探索モード時の上限
    step = 0.05                       # 自動探索モード時の刻み幅
    max_iter = 100                    # ランダム試行回数

def spatial_train_test_split(X, y, test_size = 0.7, cfg: SpatialSplitConfig = SpatialSplitConfig()):
    """
    空間ブロック分割によるtrain/test分割
    spatial_train_test_splitは以下の①、②を満たす。
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
        test_size (float, optional): 理想的なテストデータの割合(0.0~1.0)
        cfg (SpatialSplitConfig):    spatial_train_test_spalitのパラメータを簡潔に定義し、管理するためのdataclass
    Returns:
        X_train, X_test, y_train, y_test, train_mask, test_mask, best_test_size
    """
    if cfg.random_state is not None:
        np.random.seed(cfg.random_state)

    H, W = y.shape
    block_h = H // cfg.n_rows
    block_w = W // cfg.n_cols
    blocks = [(i, j) for i in range(cfg.n_rows) for j in range(cfg.n_cols)]
    total_pixels = np.sum(y != cfg.background_label)
    unique_classes = np.unique(y[y != cfg.background_label])

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
        train_mask &= (y != cfg.background_label)
        test_mask &= (y != cfg.background_label)

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
            if cls_train_count < cfg.min_train_samples_per_class:
                return None # 条件を満たさない → 再試行
            if cls_test_count < cfg.min_test_samples_per_class:
                return None # 条件を満たさない → 再試行

        return X_train, X_test, y_train, y_test, train_mask, test_mask

    # =========================================
    # cfg.auto_adjust_test_size = Falseの場合(固定)
    # =========================================

    if not cfg.auto_adjust_test_size:
        for curr in range(cfg.max_iter):
            result = try_split(test_size)
            if result is not None:
                best_test_size = len(result[1]) / total_pixels
                return (*result, best_test_size)

        raise ValueError(
            f"条件を満たす分割が見つかりませんでした。"
            f"test_size={test_size}, cfg.min_test_samples_per_class={cfg.min_test_samples_per_class}"
        )
    # =========================================
    # cfg.auto_adjust_test_size = Trueの場合(探索)
    # =========================================
    min_ratio = cfg.min_search_test_ratio or test_size
    max_ratio = cfg.max_search_test_ratio or test_size
    candidate_sizes = np.arange(min_ratio, max_ratio + 1e-8, cfg.step)

    best_result = None
    best_test_size = None
    best_diff = float("inf")

    for ts in candidate_sizes:
        for _ in range(cfg.max_iter):
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
        cfg.background_label (int): 背景クラスラベル
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
    from dataclasses import dataclass
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

    以上から、correctedは'pre-classified LULC data'に相当する可能性があると思う。
    """
    @dataclass
    class SpatialSplitConfig:
        n_rows: int = 13                       # ブロック分割数(縦)
        n_cols: int = 13                       # ブロック分割数(横)
        min_train_samples_per_class: int = 20  # 各土地被覆クラスで作成する教師データのサンプル数の下限
        min_test_samples_per_class: int = 5    # 各土地被覆クラスで作成するなテストサンプル数の下限
        background_label: int = 0              # 各土地被覆クラスで作成するなテストサンプル数の下限
        random_state: int = None               # 乱数シード
        auto_adjust_test_size: bool = False    # テストサイズを自動調整するか
        min_search_test_ratio: float = None    # 自動探索モード時の下限
        max_search_test_ratio: float = None    # 自動探索モード時の上限
        step: float = 0.05                     # 自動探索モード時の刻み幅
        max_iter: int = 100                    # ランダム試行回数

    def show_rs_info(features, gt=None, name='Dataset',cfg = SpatialSplitConfig()):
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
            # cfg.n_rows = 2,
            # cfg.n_cols = 3,
            # test_size = 0.7,
            # cfg.background_label = 0,
            # balance_classes= True,
            # target_per_class = 100,
            # cfg.random_state = 2
        # )

        # (2)spatial_train_test_split_v2での動作確認
        # test_size = 0.7
        # X_train, X_test, y_train, y_test, train_mask, test_mask = spatial_train_test_split_v2(
            # X = features,
            # y = gt,
            # cfg.n_rows = 13,
            # cfg.n_cols = 13,
            # test_size = test_size,
            # cfg.background_label = 0,
            # min_acceptable_test_ratio = test_size - 0.5,
            # max_acceptable_test_ratio = test_size + 0.5,
            # cfg.random_state = 2,
            # balance_classes = True,
            # target_per_class = 100,
            # cfg.max_iter = 100
        # )

        # (3)spatial_train_test_split_v3での動作確認
        # test_size, error_rate = 0.7, 0.05
        # X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split_v3(
            # X = features,
            # y = gt,
            # cfg.n_rows = 13,
            # cfg.n_cols = 13,
            # test_size = test_size,
            # cfg.min_search_test_ratio = test_size - error_rate,
            # cfg.max_search_test_ratio = test_size + error_rate,
            # cfg.background_label = 0,
            # cfg.random_state = None,
            # balance_classes = False,
            # target_per_class = None,
            # cfg.auto_adjust_test_size = True,
            # cfg.step = 0.05,
            # cfg.max_iter = 100
        # )

        # (4)spatial_train_test_split_v4での動作確認
        test_size, error_rate = 0.7, 0.1
        X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split(
            X = features,
            y = gt,
            test_size=test_size,
            cfg = cfg
        )

        nobg_count = gt[gt != cfg.background_label].size
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
        # cfg.background_label=0,
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
        # cfg.background_label=0,
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
        # cfg.background_label=0,
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
