from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_baseline_iA(A_vxm: np.ndarray, lambda1: float, lambda2: float):
    A = A_vxm.T.astype(np.float64, copy=False)
    ll = None
    if lambda2:
        ll0 = np.sum(A**2, axis=0, dtype=np.float64)
        ll = np.sqrt(ll0 + lambda2 * np.max(ll0))
        A = A / ll

    att = A @ A.T
    ss = np.linalg.norm(att)
    penalty = float(np.sqrt(ss) * lambda1)
    iatt = np.linalg.inv(att + (penalty**2) * np.eye(att.shape[0], dtype=np.float64))
    iA = A.T @ iatt
    if ll is not None:
        iA = iA / ll[:, None]
    return iA * (1 / 100)


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subset_npz", type=Path)
    parser.add_argument("reconstruction_py", type=Path)
    parser.add_argument("reconstruction_cached_py", type=Path)
    parser.add_argument("--lambda1", type=float, default=0.01)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--gsigma", type=float, default=3.0)
    parser.add_argument("--timepoints", type=int, default=20)
    args = parser.parse_args()

    recon = load_module("bench_recon", args.reconstruction_py)
    cached = load_module("bench_recon_cached", args.reconstruction_cached_py)

    data = np.load(args.subset_npz, allow_pickle=False)
    A_vxm = data["A_vxm"]
    good_vox = data["Good_Vox"]
    dim = {
        "nVx": int(data["nVx"][0]),
        "nVy": int(data["nVy"][0]),
        "nVz": int(data["nVz"][0]),
        "sV": float(data["sV"][0]),
        "Good_Vox": good_vox,
    }

    rng = np.random.default_rng(42)
    y = rng.normal(size=(A_vxm.shape[1], args.timepoints))

    baseline_iA, baseline_build_iA_s = timed(
        build_baseline_iA, A_vxm, args.lambda1, args.lambda2
    )
    baseline_smooth, baseline_smooth_s = timed(
        recon.smooth_Amat,
        baseline_iA,
        dim,
        args.gsigma,
    )
    baseline_final, baseline_apply_s = timed(np.matmul, baseline_smooth, y)

    cache, improved_build_s = timed(
        cached.build_tikhonov_cache_vox_by_meas,
        A_vxm,
        args.lambda1,
        args.lambda2,
        None,
        4096,
    )
    improved_img, improved_recon_s = timed(
        cached.reconstruct_img_from_cache_vox_by_meas,
        y,
        A_vxm,
        cache,
    )
    improved_smooth, improved_smooth_s = timed(
        cached.smooth_img_vox, improved_img, dim, args.gsigma
    )

    cached_img, cached_recon_s = timed(
        cached.reconstruct_img_from_cache_vox_by_meas,
        y,
        A_vxm,
        cache,
    )
    cached_smooth, cached_smooth_s = timed(
        cached.smooth_img_vox, cached_img, dim, args.gsigma
    )

    result = {
        "subset_shape": [int(A_vxm.shape[0]), int(A_vxm.shape[1])],
        "timepoints": args.timepoints,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "gsigma": args.gsigma,
        "baseline_build_iA_s": baseline_build_iA_s,
        "baseline_smooth_s": baseline_smooth_s,
        "baseline_apply_s": baseline_apply_s,
        "baseline_total_s": baseline_build_iA_s + baseline_smooth_s + baseline_apply_s,
        "improved_build_cache_s": improved_build_s,
        "improved_recon_s": improved_recon_s,
        "improved_smooth_s": improved_smooth_s,
        "improved_first_total_s": improved_build_s + improved_recon_s + improved_smooth_s,
        "improved_cached_recon_s": cached_recon_s,
        "improved_cached_smooth_s": cached_smooth_s,
        "improved_cached_total_s": cached_recon_s + cached_smooth_s,
        "baseline_vs_improved_max_abs_err": float(np.max(np.abs(baseline_final - improved_smooth))),
        "improved_repeat_max_abs_err": float(np.max(np.abs(improved_smooth - cached_smooth))),
        "speedup_first_run_vs_baseline": float(
            (baseline_build_iA_s + baseline_smooth_s + baseline_apply_s)
            / (improved_build_s + improved_recon_s + improved_smooth_s)
        ),
        "speedup_cached_vs_baseline": float(
            (baseline_build_iA_s + baseline_smooth_s + baseline_apply_s)
            / (cached_recon_s + cached_smooth_s)
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
