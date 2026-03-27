# NeuroDOT Reconstruction Benchmark Report

Date: 2026-03-27

This note compares the current explicit-`iA` reconstruction path against the
new exact dense cached reconstruction strategy on real subsets extracted from:

`/Users/amit/Downloads/A_Adult_96x92.mat`

Both paths preserve the same math:

- same `A`
- same `lambda1`
- same `lambda2`
- same Gaussian smoothing kernel

The only change is execution strategy:

- baseline: explicitly build `iA`, smooth `iA`, then apply it
- improved: build a measurement-space cache, reconstruct without materializing
  `iA`, then smooth the reconstructed image

## Benchmark Harness

Files:

- `benchmarks/benchmark_reconstruction_subset.py`
- `benchmarks/reconstruction_standalone.py`
- `benchmarks/reconstruction_cached_standalone.py`

Subset files used for the measurements were generated locally from the NITRC
matrix and were not committed.

Common parameters:

- `lambda1 = 0.01`
- `lambda2 = 0.1`
- `gsigma = 3`
- `timepoints = 20`

## Commands

Local:

```bash
python3 benchmarks/benchmark_reconstruction_subset.py \
  benchmarks/A_Adult_96x92_subset_m128.npz \
  benchmarks/reconstruction_standalone.py \
  benchmarks/reconstruction_cached_standalone.py \
  --lambda1 0.01 --lambda2 0.1 --gsigma 3 --timepoints 20

python3 benchmarks/benchmark_reconstruction_subset.py \
  benchmarks/A_Adult_96x92_subset_m256.npz \
  benchmarks/reconstruction_standalone.py \
  benchmarks/reconstruction_cached_standalone.py \
  --lambda1 0.01 --lambda2 0.1 --gsigma 3 --timepoints 20
```

br200:

```bash
ssh br200 "cd /N/u/atsubhas/NeuroDOT_bench_2026-03-27 && \
python3 benchmark_reconstruction_subset.py \
  A_Adult_96x92_subset_m128.npz \
  reconstruction_standalone.py \
  reconstruction_cached_standalone.py \
  --lambda1 0.01 --lambda2 0.1 --gsigma 3 --timepoints 20"

ssh br200 "cd /N/u/atsubhas/NeuroDOT_bench_2026-03-27 && \
python3 benchmark_reconstruction_subset.py \
  A_Adult_96x92_subset_m256.npz \
  reconstruction_standalone.py \
  reconstruction_cached_standalone.py \
  --lambda1 0.01 --lambda2 0.1 --gsigma 3 --timepoints 20"
```

## Results

### Local Mac

| Subset | Baseline Total | Improved First Run | Improved Cached Rerun | First-Run Speedup | Cached Speedup | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|
| `156461 x 128` | `0.578 s` | `0.157 s` | `0.108 s` | `3.67x` | `5.36x` | `1.90e-15` |
| `156461 x 256` | `1.295 s` | `0.261 s` | `0.132 s` | `4.96x` | `9.81x` | `2.28e-15` |

### br200

| Subset | Baseline Total | Improved First Run | Improved Cached Rerun | First-Run Speedup | Cached Speedup | Max Abs Error |
|---|---:|---:|---:|---:|---:|---:|
| `156461 x 128` | `7.079 s` | `1.078 s` | `0.392 s` | `6.56x` | `18.04x` | `1.76e-16` |
| `156461 x 256` | `14.393 s` | `1.581 s` | `0.623 s` | `9.10x` | `23.10x` | `2.57e-16` |

## Observations

- The improved path is numerically aligned with the baseline to machine
  precision on these benchmarks.
- The speedup increases with measurement count, which is what we want.
- The biggest gain comes from avoiding explicit `iA` construction and from
  smoothing the reconstructed image instead of smoothing all columns of `iA`.
- Cached reruns are materially faster than first runs because the
  measurement-space factorization is reused.

## Practical Interpretation

- The exact dense cached strategy is not just theoretically better. It is
  already significantly faster on real subsets from the actual NITRC matrix.
- On `br200`, the larger `m256` subset shows about `9.1x` first-run speedup
  and about `23.1x` cached-rerun speedup.
- This makes the approach worth integrating into the real NeuroDOT pipeline.
