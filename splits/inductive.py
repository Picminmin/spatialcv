"""
inductive評価を主張可能なtrain_test_splitの実装。
リモートセンシングデータセットを、train, test, (任意でval), unlabeledに分割する。

分割順序: train → test → (任意でval) → unlabeled
train/U は空間的に連続(長方形ブロックの連結)、val/testはtrain ∪ U と空間的に不連続にするのが基本。
unlabeledの個数の指定: trainの輪帯(近傍)から選ぶ。個数は倍率r または個数u_cntで指定。
inductiveであることを主張するため、「Uからtest/valを必ず除外」する。

実装案
train: 各クラスの最小数を満たす連結ブロック領域(長方形ブロックの集合の連結成分)をどん欲な領域拡張
で確保。
test: 残りブロックから同様に連結で各クラスの最小数を確保(目標test割合に近づける)。
val (任意): testと同様にguard帯の外から確保。
unlabeled: trainをu_ringピクセルだけ膨張した輪帯から、近さ優先でr倍 or u_cnt を切り出し
(必ず N ⊂ guard 外、test/val/背景は除外)。
動的ブロック: train_ratioが極端(小/大)なときは行列数を自動で細分化し、小さな長方形領域で取りやすくする。
"""

import numpy as np
from math import gcd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Set
from scipy.ndimage import binary_dilation, distance_transform_edt

@dataclass
class SpatialSplitConfigV2:
    # ブロック分割(基準)
    n_rows: int = 13
    n_cols: int = 13
    # 動的細分化
    dynamic_grid: bool = True
    min_rows: int = 8
    min_cols: int = 8
    max_rows: int = 26
    max_cols: int = 26
    # グリッド再試行
    grid_retry: bool = True
    grid_attempts: int = 6         # 何段階試すか
    grid_decrement_step: int = 2   # 失敗時に rows/ cols をいくつ減らすか
    try_gcd_start: bool = False    # True なら最初に GCD ベースの均等割りも試す
    # 連結成長のステップ上限 (None ならデフォルト n_blocks*2)
    grow_max_steps: Optional[int] = None
    # クラス別最小数
    min_train_per_class: int = 20
    min_test_per_class: int = 5
    min_val_per_class: int = 0 # 0 なら val なしでも可
    # 背景・乱数
    background_label: int = 0
    random_state: Optional[int] = None
    # 近傍輪帯&評価ガード
    u_ring: int = 2     # train をこのピクセルだけ膨張 → 輪帯
    max_ring: int = 10
    buffer_px: int = 6  # guard 帯の厚み(val/test は train∪U からこれだけ離す)
    # unlabeled の量 (倍率優先、していなければNone)
    unlabeled_multiplier: Optional[float] = 1.0 # |U|=ceil(r*|L|)。Noneならu_cntを使う
    unlabeled_count: Optional[int] = None
    # 反復上限
    max_iter: int = 200
    # 目標比率への許容誤差
    ratio_tol: float = 0.02 # ±2%

def _rng(seed):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()

def _make_block_ids(H:int, W:int, n_rows:int, n_cols:int) -> np.ndarray:
    """H*W を n_rows*n_cols の長方形ブロックに分け、各画素に block_id を振る"""
    row_edges = np.linspace(0, H, n_rows+1, dtype=int)
    col_edges = np.linspace(0, W, n_cols+1, dtype=int)
    block_id = np.empty((H, W), dtype = np.int32)
    bid = 0
    for i in range(n_rows):
        r0, r1 = row_edges[i], row_edges[i+1]
        for j in range(n_cols):
            c0, c1 = col_edges[j], col_edges[j+1]
            block_id[r0:r1, c0:c1] = bid
            bid += 1
    return block_id

def _class_counts_per_block(y: np.ndarray, block_id: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """shape: [n_blocks, n_classes]"""
    n_blocks = block_id.max() + 1
    cls_map = {c:i for i,c in enumerate(classes)}
    counts = np.zeros((n_blocks, len(classes)), dtype = np.int32)
    flat_bid = block_id.ravel()
    flat_y = y.ravel()
    for b, lab in zip(flat_bid, flat_y):
        if lab in cls_map:
            counts[b, cls_map[lab]] += 1
    return counts

def _adjacency(n_rows:int, n_cols:int):
    """4近傍のブロック隣接辞書"""
    def idx(i,j): return i*n_cols + j
    adj = {idx(i,j):set() for i in range(n_rows) for j in range(n_cols)}
    for i in range(n_rows):
        for j in range(n_cols):
            u = idx(i,j)
            for di,dj in ((-1,0),(1,0),(0,-1),(0,1)):
                ni, nj = i+di, j+dj
                if 0<=ni<n_rows and 0<=nj<n_cols:
                    adj[u].add(idx(ni,nj))
    return adj

def _grow_connected_blocks(counts: np.ndarray,
                           n_rows:int, n_cols:int,
                           need_per_class: np.ndarray,
                           target_pixels: Optional[int],
                           ratio_tol: float,
                           rng,
                           max_steps: Optional[int]=None) -> Optional[Set[int]]:
    """
    貪欲に連結ブロック集合を成長させ、各クラス最小数を満たし、
    可能なら総画素も target に近づける。戻り値はブロックID集合。
    """
    n_blocks, _ = counts.shape
    adj = _adjacency(n_rows, n_cols)
    # スコア: 不足分にどれだけ寄与するか
    total_per_block = counts.sum(axis = 1)
    # 初期シード: 総画素が最大のブロック
    seed = int(np.argmax(total_per_block))
    selected = {seed}
    covered = counts[seed].copy()
    # 候補境界
    frontier = set(adj[seed])

    def deficits(cov): # 各クラスの不足量
        return np.maximum(need_per_class - cov, 0)

    steps = 0
    hard_cap = max_steps if max_steps is not None else (n_blocks * 2)


    # need = need_per_class.copy()
    # iter_guard = 0
    while True:
        steps += 1

        # iter_guard += 1
        # 条件達成？
        if (covered >= need_per_class).all():
            if target_pixels is None:
                return selected
            sel_pix = total_per_block[list(selected)].sum()
            # 目標画素に十分近い(または超えた)
            if abs(sel_pix - target_pixels) <= max(1, int(ratio_tol*target_pixels)) or sel_pix >= target_pixels:
                return selected
        if not frontier or steps > hard_cap:
            # これ以上連結に拡張できない
            if (covered >= need_per_class).all():
                return selected
            return None
        # 最良ブロックを frontier から選ぶ
        def score(b):
            gain = counts[b]
            before = deficits(covered)
            after = deficits(covered + gain)
            reduced = (before - after)
            # 不足解消への貢献を主＋少し総画素も見る
            return (reduced.sum(), total_per_block[b])
        best = max(frontier, key = score)
        selected.add(best)
        covered += counts[best]
        # frontier 更新(未選択のみ)
        frontier |= adj[best]
        frontier -= selected
        # if iter_guard > n_blocks*2:
            # break
    # return None

def _mask_from_blocks(block_id: np.ndarray, blocks: Set[int]) -> np.ndarray:
    m = np.isin(block_id, list(blocks))
    return m

def _mask_unlabeled_indices(L_mask: np.ndarray,
                            background_mask: np.ndarray,
                            test_mask: np.ndarray,
                            val_mask: Optional[np.ndarray],
                            u_ring:int, max_ring:int,
                            target_count:int) -> np.ndarray:
    """train の輪帯から test/val/背景を除外し、L に近い順で target_count を切る"""
    H, W = L_mask.shape
    ex = L_mask | background_mask | test_mask
    if val_mask is not None:
        ex |= val_mask
    dist_to_L = distance_transform_edt(~L_mask).astype(np.float32)

    pool = np.zeros_like(L_mask, dtype=bool)
    ring = u_ring
    while ring  <= max_ring and pool.sum() < target_count:
        rim = binary_dilation(L_mask, iterations=ring) & (~L_mask)
        cand = rim & (~ex)
        pool |= cand
        ring += 1

    pool_idx = np.flatnonzero(pool)
    if pool_idx.size == 0:
        return pool_idx # 空

    order_key = dist_to_L.ravel()[pool_idx]
    order = np.argsort(order_key) # 近い順
    take = min(target_count, pool_idx.size)
    return pool_idx[order[:take]]

def spatial_train_test_split_v2(
    X: np.ndarray,          # (H, W, B)
    y: np.ndarray,          # (H, W)
    test_size: float,       # 0..1
    val_size: float = 0.0,  # 0..1
    cfg: SpatialSplitConfigV2 = SpatialSplitConfigV2()
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, Optional[np.ndarray],
    np.ndarray, float, float
]:
    """
    単一関数でtrain/test/(val)/unlabeled をつくる。
    Returns:
        X_train, X_test, y_train, y_test, train_mask, test_mask, val_mask(or None),
        U_index, actual_train_ratio, actual_test_ratio
    """
    if cfg.random_state is not None:
        np.random.seed(cfg.random_state)
    H, W = y.shape
    classes = np.array(sorted(c for c in np.unique(y) if c != cfg.background_label))
    if classes.size == 0:
        raise ValueError("有効クラスが見つかりません。")

    # 目標比率
    target_train_ratio = max(0.0, 1.0 - test_size - max(0.0, val_size))
    total_labeled = int(np.sum(y != cfg.background_label))
    rng = _rng(cfg.random_state)

    # --- 初期グリッド候補列をつくる ---
    # 1) dynamic_grid のヒント (極端比率なら細分化)
    nr0, nc0 = cfg.n_rows, cfg.n_cols
    if cfg.dynamic_grid and (target_train_ratio < 0.3 or target_train_ratio > 0.7):
        nr0 = min(cfg.max_rows, max(cfg.min_rows, cfg.n_rows * 2))
        nc0 = min(cfg.max_cols, max(cfg.min_cols, cfg.n_cols * 2))

    grid_candidates = []
    # 0) 任意: GCDスタート (均等割り、上限内で)
    if cfg.try_gcd_start:
        g = gcd(H, W)
        gr = min(cfg.max_rows, max(cfg.min_rows, g))
        gc = min(cfg.max_cols, max(cfg.min_cols, g))
        grid_candidates.append((gr, gc))
    # 1) 初期案
    grid_candidates.append((nr0, nc0))
    # 2) 失敗時のデクリメント
    if cfg.grid_retry:
        nr, nc = nr0, nc0
        for i in range(cfg.grid_attempts - 1):
            nr = max(cfg.min_rows, nr - cfg.grid_decrement_step)
            nc = max(cfg.min_cols, nc - cfg.grid_decrement_step)
            if (nr, nc) != grid_candidates[-1]:
                grid_candidates.append((nr, nc))
    # 重複除去 (順序保持)
    seen = set()
    grids = []
    for gpair in grid_candidates:
        if gpair not in seen:
            grids.append(gpair); seen.add(gpair)

    # ============= グリッドを変えながら試行 =============
    last_err = None
    for (nr, nc) in grids:
        try:
            block_id = _make_block_ids(H, W, nr, nc)
            n_blocks = block_id.max() + 1
            counts = _class_counts_per_block(y, block_id, classes)

            # 1) train
            need_train = np.array([cfg.min_train_per_class]*len(classes), dtype=int)
            target_train_pixels = int(target_train_ratio * total_labeled) if target_train_ratio > 0 else None
            train_blocks = _grow_connected_blocks(counts, nr, nc, need_train, target_train_pixels,
                                                  cfg.ratio_tol, rng, cfg.grow_max_steps)
            if not train_blocks:
                raise RuntimeError("train 連結選択に失敗")

            train_mask = _mask_from_blocks(block_id, train_blocks) & (y != cfg.background_label)

            # 2) test (train 以外)
            remain_blocks = set(range(n_blocks)) - set(train_blocks)
            if not remain_blocks:
                raise RuntimeError("train が全域を占有し test 不可")

            mask_remain = np.isin(block_id, list(remain_blocks))
            y_remain = np.where(mask_remain, y, cfg.background_label)
            counts_remain = _class_counts_per_block(y_remain, block_id, classes)

            need_test = np.array([cfg.min_test_per_class]*len(classes), dtype = int)
            target_test_pixels = int(test_size * total_labeled)
            test_blocks = _grow_connected_blocks(counts_remain, nr, nc, need_test, target_test_pixels,
                                                 cfg.ratio_tol, rng, cfg.grow_max_steps)
            if not test_blocks:
                raise RuntimeError("test 連結選択に失敗")
            test_blocks -= set(train_blocks)
            test_mask = _mask_from_blocks(block_id, test_blocks) & (y != cfg.background_label)

            actual_train_ratio = train_mask.sum() / max(1, total_labeled)
            actual_test_ratio = test_mask.sum() / max(1, total_labeled)

            # 3) val (任意)
            val_mask = None
            if val_size and val_size > 0.0 and cfg.min_val_per_class > 0:
                guard = binary_dilation(train_mask, iterations=cfg.buffer_px) | \
                        binary_dilation(test_mask, iterations=cfg.buffer_px)
                cand = (~guard) & (y != cfg.background_label)
                y_cand = np.where(cand, y, cfg.background_label)
                counts_cand = _class_counts_per_block(y_cand, block_id, classes)
                need_val = np.array([cfg.min_val_per_class]*len(classes), dtype = int)
                target_val_pixels = int(val_size * total_labeled)
                val_blocks = _grow_connected_blocks(counts_cand, nr, nc, need_val, target_val_pixels,
                                                    cfg.ratio_tol, rng, cfg.grow_max_steps)
                if val_blocks:
                    val_blocks -= (set(train_blocks) | set(test_blocks))
                    val_mask = _mask_from_blocks(block_id, val_blocks) & cand

            # 4) unlabeled
            background_mask = (y == cfg.background_label)
            if cfg.unlabeled_multiplier is not None:
                target_u = int(np.ceil(cfg.unlabeled_multiplier * train_mask.sum()))
            else:
                target_u = None
            if cfg.unlabeled_count is not None:
                target_u = cfg.unlabeled_count if target_u is None else min(target_u, cfg.unlabeled_count)
            if target_u in None:
                target_u = int(np.ceil(1.0 * train_mask.sum()))


            U_index = _mask_unlabeled_indices(
                L_mask = train_mask,
                background_mask = background_mask,
                test_mask = test_mask,
                val_mask = val_mask,
                u_ring = cfg.u_ring,
                max_ring = cfg.max_ring,
                target_count = target_u
            )

            # ========== 5) 出力 ==========
            X_flat = X.reshape(-1 ,X.shape[2]); y_flat = y.ravel()
            train_idx = np.flatnonzero(train_mask); test_idx = np.flatnonzero(test_mask)

            X_train, y_train = X_flat[train_idx], y_flat[train_idx]
            X_test, y_test = X_flat[test_idx], y_flat[test_idx]

            return (X_train, X_test, y_train, y_test,
                    train_mask, test_mask, val_mask,
                    U_index, actual_train_ratio, actual_test_ratio)

        except Exception as e:
            last_err = e
            # 次のグリッド候補で再試行
            continue

    # すべて失敗
    raise RuntimeError(f"グリッド再試行に失敗しました。最後のエラー: {last_err}")




if __name__ == "__main__":
    from RS_GroundTruth.rs_dataset import RemoteSensingDataset

    # データ読み込み
    ds = RemoteSensingDataset(remove_bad_bands = True)
    X, y = ds.load("Indianpines")
    H, W, B = X.shape
    # シード値
    random_state = 43
    class SpatialSplitConfigV2:
        # ブロック分割(基準)
        n_rows=13
        n_cols=13
        # 動的細分化
        dynamic_grid: bool = True
        grid_retry=True
        grid_attempts=30
        grid_decrement_step=2

        try_gcd_start=False,   # 試したいときはTrue
        grow_max_steps = 2000  # 連結拡張の上限。重いときは 2000 など指定
        min_rows: int = 8
        min_cols: int = 8
        max_rows: int = 26
        max_cols: int = 26
        # クラス別最小数
        min_train_per_class: int = 10
        min_test_per_class: int = 5
        min_val_per_class: int = 0 # 0 なら val なしでも可
        # 背景・乱数
        background_label: int = 0
        random_state: Optional[int] = random_state
        # 近傍輪帯&評価ガード
        u_ring: int = 2     # train をこのピクセルだけ膨張 → 輪帯
        max_ring: int = 10
        buffer_px: int = 6  # guard 帯の厚み(val/test は train∪U からこれだけ離す)
        # unlabeled の量 (倍率優先、していなければNone)
        unlabeled_multiplier: Optional[float] = 1.0 # |U|=ceil(r*|L|)。Noneならu_cntを使う
        unlabeled_count: Optional[int] = None
        # 反復上限
        max_iter: int = 500
        # 目標比率への許容誤差
        ratio_tol: float = 0.02 # ±2%

    cfg = SpatialSplitConfigV2()
    X_train, X_test, y_train, y_test, train_mask, test_mask, val_mask, U_index, tr_r, te_r = \
        spatial_train_test_split_v2(X, y, test_size = 0.4, val_size = 0.1, cfg=cfg)

    print(f"|L|={train_mask.sum()}, |U|={len(U_index)}, |T|={test_mask.sum()}  train={tr_r:.3f}, test={te_r:.3f}")
