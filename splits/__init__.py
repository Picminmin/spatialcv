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
                           rng) -> Optional[Set[int]]:
    """
    貪欲に連結ブロック集合を成長させ、各クラス最小数を満たし、
    可能なら総画素も target に近づける。戻り値はブロックID集合。
    """
    n_blocks, n_cls = counts.shape
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

    need = need_per_class.copy()
    iter_guard = 0
    while True:
        iter_guard += 1
        # 条件達成？
        if (covered >= need).all():
            if target_pixels is None:
                return selected
            sel_pix = total_per_block[list(selected)].sum()
            # 目標画素に十分近い(または超えた)
            if abs(sel_pix - target_pixels) <= max(1, int(ratio_tol*target_pixels)) or sel_pix >= target_pixels:
                return selected
        if not frontier:
            # これ以上連結に拡張できない
            if (covered >= need).all():
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
        if iter_guard > n_blocks*2:
            break
    return None

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
