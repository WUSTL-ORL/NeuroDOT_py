from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def smooth_Amat(iA_in, dim, gsigma):
    nvox = np.shape(iA_in)[0]
    nm = np.shape(iA_in)[1]
    iA_out = np.zeros((nvox, nm), dtype=np.float64)

    nVx = int(dim["nVx"])
    nVy = int(dim["nVy"])
    nVz = int(dim["nVz"])
    gsigma = gsigma / dim["sV"]

    if "Good_Vox" in dim:
        gv = np.asarray(dim["Good_Vox"], dtype=np.int64).reshape(-1)
    else:
        gv = np.arange(1, nVx * nVy * nVz + 1, dtype=np.int64)

    for k in range(nm):
        iAvox = np.zeros((nVx * nVy * nVz,), dtype=np.float64)
        iAvox[gv - 1] = iA_in[:, k]
        iAvox = np.reshape(iAvox, (nVx, nVy, nVz), order="F")
        iAvox = ndi.gaussian_filter(
            iAvox,
            gsigma,
            mode="nearest",
            truncate=np.ceil(2 * gsigma) / gsigma,
        )
        iA_out[:, k] = np.reshape(iAvox, (-1,), order="F")[gv - 1]

    return iA_out
