
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
