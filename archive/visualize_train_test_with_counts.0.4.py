
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

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
