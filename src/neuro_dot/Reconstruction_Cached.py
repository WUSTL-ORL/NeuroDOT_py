from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.linalg as spla
import scipy.ndimage as ndi


@dataclass
class TikhonovCache:
    """Cached measurement-space solve state for an exact dense reconstruction."""

    cho: tuple[np.ndarray, bool]
    keep_idx: np.ndarray
    lambda1: float
    lambda2: float
    penalty: float
    ll: Optional[np.ndarray]
    gram: np.ndarray
    block_rows: int
    dtype: np.dtype


def _normalize_keep(
    measurement_count: int, keep: Optional[np.ndarray]
) -> np.ndarray:
    if keep is None:
        return np.arange(measurement_count, dtype=np.int64)

    keep_arr = np.asarray(keep)
    if keep_arr.dtype == np.bool_:
        if keep_arr.ndim != 1 or keep_arr.shape[0] != measurement_count:
            raise ValueError("Boolean keep mask must match measurement count.")
        return np.flatnonzero(keep_arr)

    keep_idx = np.asarray(keep_arr, dtype=np.int64).reshape(-1)
    if keep_idx.size == 0:
        raise ValueError("Keep mask produced no measurements.")
    if np.any(keep_idx < 0) or np.any(keep_idx >= measurement_count):
        raise ValueError("Keep indices are out of bounds.")
    return np.unique(keep_idx)


def _iter_voxel_blocks(
    A_vxm: Any, keep_idx: np.ndarray, block_rows: int
):
    voxel_count = int(A_vxm.shape[0])
    for start in range(0, voxel_count, block_rows):
        stop = min(start + block_rows, voxel_count)
        block = np.asarray(A_vxm[start:stop, keep_idx], dtype=np.float64)
        yield start, stop, block


def compute_svr_weights_vox_by_meas(
    A_vxm: Any,
    keep: Optional[np.ndarray] = None,
    lambda2: float = 0.0,
    block_rows: int = 2048,
) -> Optional[np.ndarray]:
    """
    Compute the exact NeuroDOT spatially variant regularization weights.
    """
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
    ll: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Build the exact measurement-space Gram matrix for a dense voxel x measurement A.
    """
    measurement_count = int(A_vxm.shape[1])
    keep_idx = _normalize_keep(measurement_count, keep)
    ll_vec = ll
    if lambda2 and ll_vec is None:
        ll_vec = compute_svr_weights_vox_by_meas(
            A_vxm, keep=keep_idx, lambda2=lambda2, block_rows=block_rows
        )

    gram = np.zeros((keep_idx.size, keep_idx.size), dtype=np.float64)
    for start, stop, block in _iter_voxel_blocks(A_vxm, keep_idx, block_rows):
        if ll_vec is not None:
            block = block / ll_vec[start:stop, None]
        gram += block.T @ block

    return gram, ll_vec, keep_idx


def subset_measurement_gram(
    full_gram: np.ndarray, keep: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    keep_idx = _normalize_keep(int(full_gram.shape[0]), keep)
    return full_gram[np.ix_(keep_idx, keep_idx)], keep_idx


def factorize_tikhonov_gram(
    gram: np.ndarray,
    lambda1: float,
) -> tuple[tuple[np.ndarray, bool], float]:
    """
    Factor the NeuroDOT measurement-space Tikhonov system.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("Gram matrix must be square.")

    ss = np.linalg.norm(gram)
    penalty = float(np.sqrt(ss) * float(lambda1))
    regularized = gram + (penalty**2) * np.eye(gram.shape[0], dtype=np.float64)
    return spla.cho_factor(regularized, overwrite_a=False, check_finite=False), penalty


def build_tikhonov_cache_vox_by_meas(
    A_vxm: Any,
    lambda1: float,
    lambda2: float = 0.0,
    keep: Optional[np.ndarray] = None,
    block_rows: int = 2048,
    full_gram: Optional[np.ndarray] = None,
) -> TikhonovCache:
    """
    Build an exact cached solve state for dense voxel x measurement A-matrices.

    When lambda2 is zero, callers may pass a precomputed full measurement-space
    Gram matrix and reuse principal submatrices across keep masks.
    """
    if lambda2 and full_gram is not None:
        raise ValueError("full_gram reuse is only exact when lambda2 == 0.")

    if full_gram is not None:
        gram, keep_idx = subset_measurement_gram(full_gram, keep)
        ll = None
    else:
        gram, ll, keep_idx = build_measurement_gram_vox_by_meas(
            A_vxm,
            keep=keep,
            lambda2=lambda2,
            block_rows=block_rows,
        )

    cho, penalty = factorize_tikhonov_gram(gram, lambda1=lambda1)
    return TikhonovCache(
        cho=cho,
        keep_idx=keep_idx,
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        penalty=penalty,
        ll=ll,
        gram=gram,
        block_rows=int(block_rows),
        dtype=np.dtype(np.float64),
    )


def build_measurement_gram_from_mat_file(
    mat_path: str | Path,
    dataset_name: str = "A",
    keep: Optional[np.ndarray] = None,
    lambda2: float = 0.0,
    block_rows: int = 2048,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Open a MATLAB v7.3 file lazily and build an exact measurement-space Gram.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        return build_measurement_gram_vox_by_meas(
            f[dataset_name],
            keep=keep,
            lambda2=lambda2,
            block_rows=block_rows,
        )


def build_tikhonov_cache_from_mat_file(
    mat_path: str | Path,
    lambda1: float,
    lambda2: float = 0.0,
    keep: Optional[np.ndarray] = None,
    block_rows: int = 2048,
    dataset_name: str = "A",
    full_gram: Optional[np.ndarray] = None,
) -> TikhonovCache:
    """
    Open a MATLAB v7.3 file lazily and build an exact cached solve state.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        return build_tikhonov_cache_vox_by_meas(
            f[dataset_name],
            lambda1=lambda1,
            lambda2=lambda2,
            keep=keep,
            block_rows=block_rows,
            full_gram=full_gram,
        )


def reconstruct_img_from_cache_vox_by_meas(
    lmdata: np.ndarray,
    A_vxm: Any,
    cache: TikhonovCache,
    units_scaling: float = 1 / 100,
    out_dtype: Any = np.float64,
) -> np.ndarray:
    """
    Reconstruct images without materializing iA.
    """
    y = np.asarray(lmdata, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != cache.keep_idx.size:
        raise ValueError("lmdata row count must match cached keep mask size.")

    z = spla.cho_solve(cache.cho, y, check_finite=False)
    voxel_count = int(A_vxm.shape[0])
    img = np.zeros((voxel_count, y.shape[1]), dtype=np.float64)

    for start, stop, block in _iter_voxel_blocks(
        A_vxm, cache.keep_idx, cache.block_rows
    ):
        if cache.ll is not None:
            block = block / cache.ll[start:stop, None]
        img_block = block @ z
        if cache.ll is not None:
            img_block = img_block / cache.ll[start:stop, None]
        img[start:stop, :] = img_block

    img *= float(units_scaling)
    return np.asarray(img, dtype=out_dtype)


def reconstruct_img_from_mat_file(
    lmdata: np.ndarray,
    mat_path: str | Path,
    cache: TikhonovCache,
    dataset_name: str = "A",
    units_scaling: float = 1 / 100,
    out_dtype: Any = np.float64,
) -> np.ndarray:
    """
    Open a MATLAB v7.3 file lazily and reconstruct without materializing iA.
    """
    import h5py

    with h5py.File(mat_path, "r") as f:
        return reconstruct_img_from_cache_vox_by_meas(
            lmdata,
            f[dataset_name],
            cache,
            units_scaling=units_scaling,
            out_dtype=out_dtype,
        )


def smooth_img_vox(
    img_in: np.ndarray,
    dim: dict,
    gsigma: float,
    out_dtype: Any = np.float64,
) -> np.ndarray:
    """
    Apply the same Gaussian voxel-space smoothing to reconstructed images.

    This is the image-space equivalent of smoothing iA before reconstruction.
    """
    img = np.asarray(img_in, dtype=np.float64)
    if img.ndim == 1:
        img = img[:, None]

    nVx = int(dim["nVx"])
    nVy = int(dim["nVy"])
    nVz = int(dim["nVz"])
    sigma = float(gsigma) / float(dim["sV"])

    if "Good_Vox" in dim:
        gv = np.asarray(dim["Good_Vox"], dtype=np.int64).reshape(-1) - 1
    else:
        gv = np.arange(nVx * nVy * nVz, dtype=np.int64)

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

    return np.asarray(smoothed, dtype=out_dtype)


def save_measurement_gram_cache(
    cache_path: str | Path,
    gram: np.ndarray,
    keep_idx: Optional[np.ndarray] = None,
    ll: Optional[np.ndarray] = None,
) -> None:
    """
    Persist a measurement-space Gram cache to disk.
    """
    payload = {"gram": np.asarray(gram, dtype=np.float64)}
    if keep_idx is not None:
        payload["keep_idx"] = np.asarray(keep_idx, dtype=np.int64)
    if ll is not None:
        payload["ll"] = np.asarray(ll, dtype=np.float64)
    np.savez_compressed(cache_path, **payload)


def load_measurement_gram_cache(
    cache_path: str | Path,
) -> dict[str, np.ndarray]:
    """
    Load a measurement-space Gram cache from disk.
    """
    with np.load(cache_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
