from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.linalg as spla
import scipy.ndimage as ndi


@dataclass
class TikhonovCache:
    cho: tuple[np.ndarray, bool]
    keep_idx: np.ndarray
    lambda1: float
    lambda2: float
    penalty: float
    ll: Optional[np.ndarray]
    gram: np.ndarray
    block_rows: int


def _normalize_keep(measurement_count: int, keep: Optional[np.ndarray]) -> np.ndarray:
    if keep is None:
        return np.arange(measurement_count, dtype=np.int64)

    keep_arr = np.asarray(keep)
    if keep_arr.dtype == np.bool_:
        return np.flatnonzero(keep_arr)
    keep_idx = np.asarray(keep_arr, dtype=np.int64).reshape(-1)
    return np.unique(keep_idx)


def _iter_voxel_blocks(A_vxm: Any, keep_idx: np.ndarray, block_rows: int):
    voxel_count = int(A_vxm.shape[0])
    for start in range(0, voxel_count, block_rows):
        stop = min(start + block_rows, voxel_count)
        yield start, stop, np.asarray(A_vxm[start:stop, keep_idx], dtype=np.float64)


def compute_svr_weights_vox_by_meas(
    A_vxm: Any,
    keep: Optional[np.ndarray] = None,
    lambda2: float = 0.0,
    block_rows: int = 2048,
) -> Optional[np.ndarray]:
    if not lambda2:
        return None

    keep_idx = _normalize_keep(int(A_vxm.shape[1]), keep)
    ll_0 = np.empty(int(A_vxm.shape[0]), dtype=np.float64)
    for start, stop, block in _iter_voxel_blocks(A_vxm, keep_idx, block_rows):
        ll_0[start:stop] = np.sum(block * block, axis=1, dtype=np.float64)
    ll = np.sqrt(ll_0 + float(lambda2) * float(np.max(ll_0)))
    ll[ll == 0] = 1.0
    return ll


def build_measurement_gram_vox_by_meas(
    A_vxm: Any,
    keep: Optional[np.ndarray] = None,
    lambda2: float = 0.0,
    block_rows: int = 2048,
):
    measurement_count = int(A_vxm.shape[1])
    keep_idx = _normalize_keep(measurement_count, keep)
    ll = compute_svr_weights_vox_by_meas(A_vxm, keep=keep_idx, lambda2=lambda2, block_rows=block_rows)
    gram = np.zeros((keep_idx.size, keep_idx.size), dtype=np.float64)
    for start, stop, block in _iter_voxel_blocks(A_vxm, keep_idx, block_rows):
        if ll is not None:
            block = block / ll[start:stop, None]
        gram += block.T @ block
    return gram, ll, keep_idx


def build_tikhonov_cache_vox_by_meas(
    A_vxm: Any,
    lambda1: float,
    lambda2: float = 0.0,
    keep: Optional[np.ndarray] = None,
    block_rows: int = 2048,
) -> TikhonovCache:
    gram, ll, keep_idx = build_measurement_gram_vox_by_meas(
        A_vxm,
        keep=keep,
        lambda2=lambda2,
        block_rows=block_rows,
    )
    ss = np.linalg.norm(gram)
    penalty = float(np.sqrt(ss) * float(lambda1))
    regularized = gram + (penalty**2) * np.eye(gram.shape[0], dtype=np.float64)
    cho = spla.cho_factor(regularized, overwrite_a=False, check_finite=False)
    return TikhonovCache(cho, keep_idx, float(lambda1), float(lambda2), penalty, ll, gram, int(block_rows))


def reconstruct_img_from_cache_vox_by_meas(
    lmdata: np.ndarray,
    A_vxm: Any,
    cache: TikhonovCache,
    units_scaling: float = 1 / 100,
) -> np.ndarray:
    y = np.asarray(lmdata, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]

    z = spla.cho_solve(cache.cho, y, check_finite=False)
    voxel_count = int(A_vxm.shape[0])
    img = np.zeros((voxel_count, y.shape[1]), dtype=np.float64)

    for start, stop, block in _iter_voxel_blocks(A_vxm, cache.keep_idx, cache.block_rows):
        if cache.ll is not None:
            block = block / cache.ll[start:stop, None]
        img_block = block @ z
        if cache.ll is not None:
            img_block = img_block / cache.ll[start:stop, None]
        img[start:stop, :] = img_block

    return img * float(units_scaling)


def smooth_img_vox(img_in: np.ndarray, dim: dict, gsigma: float) -> np.ndarray:
    img = np.asarray(img_in, dtype=np.float64)
    if img.ndim == 1:
        img = img[:, None]

    nVx = int(dim["nVx"])
    nVy = int(dim["nVy"])
    nVz = int(dim["nVz"])
    sigma = float(gsigma) / float(dim["sV"])
    gv = np.asarray(dim["Good_Vox"], dtype=np.int64).reshape(-1) - 1

    smoothed = np.zeros_like(img)
    for t in range(img.shape[1]):
        vol = np.zeros(nVx * nVy * nVz, dtype=np.float64)
        vol[gv] = img[:, t]
        vol = np.reshape(vol, (nVx, nVy, nVz), order="F")
        vol = ndi.gaussian_filter(
            vol,
            sigma,
            mode="nearest",
            truncate=np.ceil(2 * sigma) / sigma,
        )
        smoothed[:, t] = np.reshape(vol, (-1,), order="F")[gv]
    return smoothed
