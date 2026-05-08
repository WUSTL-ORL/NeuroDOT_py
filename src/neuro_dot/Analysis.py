# General imports
import numpy as np
import numpy.linalg as lna
import neuro_dot as ndot
from scipy import signal
from scipy.linalg import pinv
from scipy.stats import zscore as scipy_zscore
import plotly.io as pio
import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots





def plot_censoring(DM, keep, GVTD, th, info, Nt):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available – skipping censoring plot.")
        return

    # Normalise DM for display (optional, but often helps)
    DM_min = DM.min()
    DM_max = DM.max()
    DM_norm = (DM - DM_min) / (DM_max - DM_min + 1e-12)

    # Keep column: 1 for kept, 0 for censored
    keep_col = keep.astype(float)

    # Add keep as an extra column
    combined = np.column_stack([DM_norm, keep_col])  # shape: (Nt, n_reg+1)

    # --- Layout: tall DM on the left, two panels on the right ---
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Left: design matrix + keep, spanning both rows (0:2, 0)
    ax_dm = fig.add_subplot(gs[:, 0])
    # Show time on x-axis, regressors on y-axis => no transpose
    im = ax_dm.imshow(combined, aspect='auto', cmap='gray',
                      extent=[0, Nt, combined.shape[1], 0],interpolation = 'nearest')
    ax_dm.imshow(combined, aspect='auto', cmap='gray',interpolation='nearest')
    ax_dm.set_xlabel('Regressors + keep')
    ax_dm.set_ylabel('Time (samples)')
    ax_dm.set_title('Design Matrix + keep')

    # Top-right: GVTD time trace
    ax_gvtd = fig.add_subplot(gs[0, 1])
    ax_gvtd.plot(np.arange(Nt), GVTD, 'k')
    ax_gvtd.axhline(th, color='r')
    ax_gvtd.set_xlim([0, Nt])
    ax_gvtd.set_ylabel('GVTD')
    ax_gvtd.set_xlabel('Time (samples)')

    # Bottom-right: GVTD histogram
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.hist(GVTD.ravel(), bins=100)
    ylim = ax_hist.get_ylim()
    ax_hist.plot([th, th], [0, ylim[1]], 'r')
    ax_hist.set_xlabel('GVTD')
    ax_hist.set_ylabel('Count')

    plt.tight_layout()
    plt.show()
   
   
def BlockAverage(data_in, pulse, dt, Tkeep = 0):
    """ 
    BLOCKAVERAGE Averages data by stimulus blocks.

    data_out = BLOCKAVERAGE(data_in, pulse, dt) takes a data array "data_in" 
    and uses the pulse and dt information to cut that data timewise into 
    blocks of equal length (dt), which are then averaged together and 
    output as "data_out".
    
    Tkeep is a temporal mask. Any time points with a zero in this vector is
    set to NaN.
    """
    ## Parameters and Initialization.
    dims = np.shape(data_in)
    Nt = dims[-1]
    NDtf = np.ndim(data_in) > 2
    Nbl = len(pulse)

    if Tkeep == 0:
        Tkeep = np.ones(shape = (Nt, 1))==1 

    # Check to make sure that the block after the last synch point for this
    if (dt + pulse[-1] - 1) > Nt:
        Nbl = Nbl - 1

    ## N-D Input (for 3-D or N-D voxel spaces).
    if NDtf:
        data_in = np.reshape(data_in, [], Nt)

    ## Incorporate Tkeep  
    data_in[:, np.argwhere(Tkeep == False)] = np.nan

    ## Cut data into blocks.
    Nm = np.shape(data_in)[0]
    blocks = np.zeros((Nm, dt, Nbl))

    for p in range(0,len(pulse)):
        pulse[p] = pulse[p] +1

    for k in range(0, Nbl):
        pulse_k = int(pulse[k])

        if (pulse[k] + dt -1) <= Nt:
            blocks[:, :, k] = data_in[:, pulse_k-1:pulse_k + dt-1] #Need to subtract 1 from both indices to account for 0 indexing in python
        else:
            dtb = (pulse_k-1) + dt - 1 - Nt+1
            nans = np.empty(shape = (np.shape(data_in)[0], dtb)) # need multiple lines to create an array of nans in python
            nans[:] = np.NaN
            blocks[:, :, k] = np.concatenate((data_in[:, (pulse_k-1):Nt], nans), axis = 1) # need to subtract 1 from start index: pulse[k] due to zero indexing, also need to use Nt as final index to get correct size

    ## Average blocks and return.
    BA_out = np.nanmean(blocks, axis = 2) 
    BSTD_out = np.nanstd(blocks, axis = 2, ddof = 1) 
    nanmean_cols = np.nanmean(BA_out, axis = 1)
    nanmean_matrix = np.ones(np.shape(BA_out))
    for x in range(0, dt): 
        nanmean_matrix[:, x] = nanmean_cols
    BA_out = BA_out - nanmean_matrix
    BT_out = np.divide(BA_out, BSTD_out)
    BT_out[np.argwhere(np.isinf(BT_out))] = 0

    ## N-D Output.
    if NDtf:
        #create tuple containing the desired output shape for BA_out, BSTD_out and BT_out
        #tuple contains first axis until second to last axis of data with dt appeneded to the end of it
        newshape = tuple(np.append(np.array(dims[0:-1]), dt))
        BA_out = np.reshape(BA_out, newshape)
        BSTD_out = np.reshape(BSTD_out, newshape)
        BT_out = np.reshape(BT_out, newshape)
        newshape_blocks = tuple(np.append(np.array(dims[0:-1]), (dt, Nbl))) # create tuple for deisred output shape for blocks, different from previous newshape bc Nbl is also appended
        blocks = np.reshape(blocks, newshape_blocks)


    return BA_out, BSTD_out, BT_out, blocks


def CalcGVTD(data):
    """
    CalcGVTD calculates the Root Mean Square across measurements (log-mean light levels or voxels) of the temporal derivative. 

    The data is assumed to have measurements in the first and time in the last 
    dimension. 
    
    Any selection of measurement type or voxel index must be done
    outside of this function.
    """
    # Double check data has correct dimensions
    # Dsizes=size(data);
    Dsizes = np.shape(data)
    Ndim = len(Dsizes)
    if Ndim > 2:
        data = np.reshape(data, [], Dsizes[-1])
    
    # 1st Temporal Derivative
    Ddata = data - np.roll(data, np.array([0, -1]), np.array([0,-1])) 

    # RMS across measurements
    GVTD = np.concatenate(([0], ndot.rms_py(Ddata[:,0:-1])), axis = 0)

    return GVTD


def FindGoodMeas(data, info_in, bthresh = 0.075):

    """ 
    FINDGOODMEAS Performs "Good Measurements" analysis to return indices of measurements within a chosen threshold.

    info_out = FINDGOODMEAS(data, info_in) takes a light-level array "data"
    in the MEAS x TIME format, and calculates the std of each channel
    as its noise level. 

    These are then thresholded by the default value of
    0.075 to create a logical array, and both are returned as MEAS x 1
    columns of the "info.MEAS" table. 
    
    If pulse synch point information exists in "info.system.synchpts",
    then FINDGOODMEAS will crop the data to the start and stop pulses.

    info_out = FINDGOODMEAS(data, info_in, bthresh) allows the user to
    specify a threshold value.
    
    See Also: PLOTCAPGOODMEAS, PLOTHISTOGRAMSTD.
    """
    ## Parameters and Initialization.
    # look for required info, FGM will not run if these fields are nonexistant
    try:
        info1 = info_in['system']['framerate']
    except KeyError:
        print('info_in["system"]["framerate"] does not exist and is required')
        print('exiting FindGoodMeas')
        return()
    try:
        info1 = info_in['pairs']
    except KeyError:
        print('info_in["pairs"] does not exist and is required')
        print('exiting FindGoodMeas')
        return()
    info_out = info_in.copy() # create info_out
    try:
        GVwin
    except NameError:
        GVwin = 600
    if not 'paradigm' in info_out:
        info_out['paradigm'] = {}
    if not bthresh in locals():
        bthresh = 0.075 # Empirically derived threshold value.
    dims = data.shape
    Nt = dims[-1]       # Assumes time is always the last dimension
    NDtf = np.ndim(data) > 2
    if GVwin > (Nt-1):
        GVwin = (Nt-1)
    
    # N-D Input.
    if NDtf:
        data = np.reshape(data, [], Nt)
    
    # Crop data to synchpts if necessary. 
    keep = np.logical_and(info_in['pairs']['r2d'] < 20, info_in['pairs']['WL'] == 2)
    foo = np.squeeze(data[keep,:])
    foo = ndot.highpass(foo, 0.02, info_in['system']['framerate']) # bandpass filter, omega_hp = 0.02
    foo = ndot.lowpass(foo, 1, info_in['system']['framerate'])     # bandpass filter, omega_lp = 1
    foo = foo - np.roll(foo, 1, 1)
    foo[:,0] = 0
    foob = ndot.rms_py(foo) # uses new rms that only takes one input, calculates rms for every column in rms_input and outputs row vector
    NtGV = Nt - GVwin
    NtGV_mat = np.ones((1,NtGV), dtype = np.int8)  

    if NtGV > 1: # sliding window to grab a meaningful set for 'quiet'
        GVTD_win_means = np.zeros(NtGV, order = 'F')
        i = 0
        while i <= (NtGV - 1):
            GVTD_win_means[i] =  np.mean(foob[i:((i+1)+ GVwin - 1)])
            i = i+1
        t0 = np.where(GVTD_win_means == np.min(GVTD_win_means)) # find min and set t0 --> tF
        tF = t0[0][0] + GVwin 
        STD = np.std(data[:, t0[0][0]:tF], 1, ddof= 1)          # Calulate STD, make sure ddof param is set = 1 so that np.STD behaves the same as matlab STD
    elif not 'synchpts' in info_out['paradigm']:
        NsynchPts = len(info_out['paradigm']['synchpts'])
        if NsynchPts > 2:
            tF = info_out['paradigm']['synchpts'][-1]
            t0 = info_out['paradigm']['synchpts'][0]
        elif NsynchPts == 2:
            tF = info_out['paradigm']['synchpts'][1]
            t0 = info_out['paradigm']['synchpts'][0]
        else:
            tF = data.shape[1]
        STD = np.std(data[:, t0:tF], 1, ddof=1)                 # Calculate STD.
    else:
        STD = np.std(data, 1, ddof=1)
    
    # Populate in table of on-the-fly calculated stuff.
    info_out['GVTDparams'] = {}
    info_out['GVTDparams']['t0'] = t0[0][0]
    info_out['GVTDparams']['tF'] = tF
    if not 'MEAS' in info_out:
        info_out['MEAS'] = {}
        info_out['MEAS']['STD'] = STD
        info_out['MEAS']['GI'] = np.zeros(np.shape(STD), dtype = np.uint8)
        info_out['MEAS']['GI'][np.where(STD <= bthresh)] = 1
    else:
        info_out['MEAS']['STD'] = STD
        info_out['MEAS']['GI'] = np.zeros(np.shape(STD), dtype = np.uint8)
        info_out['MEAS']['GI'][np.where(STD <= bthresh)] = 1
    if 'Clipped' in info_out['MEAS']:
        info_out['MEAS']['GI'] = np.zeros(np.shape(STD), dtype = np.uint8)
        info_out['MEAS']['GI'][np.where(info_out['MEAS']['GI'] and not info_out['MEAS']['Clipped'])] = 1
    
    return info_out


def fisher_r2z(r):
    """Apply Fisher R-to-Z (arctanh) transformation, clipping to avoid infinities."""
    return np.arctanh(np.clip(r, -0.9999999, 0.9999999))


def GLM(data, hrf, info, params=None):
    """
    General Linear Model (GLM) based on a hemodynamic response function (HRF)
    and stimulus times (info['paradigm']['synchpts']).

    Parameters
    ----------
    data   : ndarray, shape (voxels, time)
    hrf    : ndarray, 1-D HRF kernel
    info   : dict  with keys like 'paradigm', 'GVTD_filt_rs', 'system', etc.
    params : dict  of options (see defaults below)

    Returns
    -------
    b      : ndarray (voxels, regressors)  : beta map
    e      : ndarray (voxels, time)        : residuals
    DM     : ndarray (time, regressors)    : design matrix (after convolution)
    EDM    : ndarray (time, events)        : experimental design matrix
    keep   : ndarray (bool) or None        : timepoint mask (GVTD censoring)
    r_sqrd : ndarray (voxels,)             : R² per voxel
    """

    # ── Defaults ─────────────────────────────────────────────────────────────
    if params is None:
        params = {}

    p = {
        'type':                 'events',
        'GVTDreg':              False,
        'Nuisance_Reg':         False,
        'Nuisance_Regs':        None,
        'GVTD_censor':          False,
        'GVTD_censor_post_win': 0,
        'GVTD_censor_pre_win':  0,
        'GVTD_th':              1e-3,
        'DoFilter':             False,
        'zscore':               True,
        'zscoreData':           False,
        'MakePlot':             True,
    }
    p.update(params)

    Nt = data.shape[1]
    EL = p.get('event_length', 0)

    paradigm  = info['paradigm']
    synchpts  = paradigm['synchpts']   # 1-D array of sync point indices
    synchtype = paradigm['synchtype']  # 1-D array of event type labels
    Nevents   = len(synchpts)

    if 'events' in p:
        events        = np.unique(p['events'])
        NuniqueEvents = len(events)
    else:
        NuniqueEvents = len(np.unique(synchtype))
        events        = np.arange(1, NuniqueEvents + 1)

    # ── Filter parameters ─────────────────────────────────────────────────────
    if p['DoFilter']:
        flags = info.get('flags', {})

        omega_hp = p.get('omega_hp',
                   flags.get('omega_hp', 0.02))

        if 'omega_lp' in p:
            omega_lp = p['omega_lp']
        elif 'omega_lp1' in flags:
            omega_lp = flags['omega_lp1']
        elif flags.get('lowpass3') and 'omega_lp3' in flags:
            omega_lp = flags['omega_lp3']
        elif flags.get('lowpass2') and 'omega_lp2' in flags:
            omega_lp = flags['omega_lp2']
        elif flags.get('lowpass1') and 'omega_lp1' in flags:
            omega_lp = flags['omega_lp1']
        else:
            omega_lp = 0.5

        system = info.get('system', {})
        fr = system.get('framerate', info.get('framerate', 1.0))

        if omega_lp > fr / 2:
            omega_lp = fr / 2
        if omega_lp == fr / 2:
            omega_lp *= 0.95

    # ── GVTD ─────────────────────────────────────────────────────────────────
    GVTD = None
    if 'GVTD_filt_rs' in info:
        GVTD = info['GVTD_filt_rs']
    elif 'misc' in info and 'GVTD_filt_rs' in info['misc']:
        GVTD = info['misc']['GVTD_filt_rs']
    elif 'GVTD' in p:
        GVTD = p['GVTD']

    # ── Experimental Design Matrix (EDM) ──────────────────────────────────────
    EDM = np.zeros((Nt, Nevents))
    for j in range(Nevents - 1):
        start = synchpts[j]
        if EL:
            end = min(synchpts[j] + EL, Nt)   # inclusive start, exclusive end
        else:
            end = synchpts[j + 1]              # up to (not including) next event
        EDM[start:end, j] = 1

    EDM[synchpts[-1]:, Nevents - 1] = 1

    if p['type'] == 'events':
        EDMu = np.zeros((Nt, NuniqueEvents))
        for j, ev in enumerate(events):
            pulse_key = f'Pulse_{int(ev)}'
            pulse_indices = np.array(paradigm[pulse_key]).ravel().astype(int) - 1  # convert to 0-based
            EDMu[:, j] = EDM[:, pulse_indices].sum(axis=1)
        EDM = EDMu

    # ── Convolve EDM with HRF ────────────────────────────────────────────────
    hrf_col = hrf.ravel()
    # conv2 in MATLAB pads in time; replicate with 1-D convolution per column
    DM = np.column_stack([
        np.convolve(EDM[:, col], hrf_col)[:Nt]
        for col in range(EDM.shape[1])
    ])

    # ── Add nuisance regressors ───────────────────────────────────────────────
    if p['GVTDreg'] and GVTD is not None:
        DM = np.column_stack([DM, GVTD.ravel()])
    if p['Nuisance_Reg'] and p['Nuisance_Regs'] is not None:
        DM = np.column_stack([DM, p['Nuisance_Regs']])

    # ── Standardise ───────────────────────────────────────────────────────────
    if p['zscore']:
        DM = scipy_zscore(DM, axis=0)           # standardise each column
    else:
        DM = DM - DM.mean(axis=0)               # mean-subtract each column

    if p['zscoreData']:
        data = scipy_zscore(data, axis=1)

    # ── Filter DM ────────────────────────────────────────────────────────────
    if p['DoFilter']:
        # detrend_tts equivalent: linear detrend along time axis
        DM = signal.detrend(DM, axis=0)

        # Butterworth high-pass then low-pass (zero-phase via filtfilt)
        sos_hp = signal.butter(4, omega_hp / (fr / 2), btype='high', output='sos')
        sos_lp = signal.butter(4, omega_lp / (fr / 2), btype='low',  output='sos')
        DM = signal.sosfiltfilt(sos_hp, DM, axis=0)
        DM = signal.sosfiltfilt(sos_lp, DM, axis=0)

    # ── Add intercept ────────────────────────────────────────────────────────
    DM = np.column_stack([np.ones(Nt), DM])

    # ── GVTD censoring ───────────────────────────────────────────────────────
    keep = None
    if p['GVTD_censor'] and GVTD is not None:
        keep = GVTD < p['GVTD_th']

        if p['GVTD_censor_pre_win']:
            keep0 = keep.copy()
            for n in range(p['GVTD_censor_pre_win'], 0, -1):
                keep = keep & np.roll(keep0, -n)

        if p['GVTD_censor_post_win']:
            keep0 = keep.copy()
            for n in range(1, p['GVTD_censor_post_win'] + 1):
                keep = keep & np.roll(keep0, n)

        if p['MakePlot']:
            ndot.plot_censoring(DM, keep, GVTD, p['GVTD_th'], info, Nt)

        DM   = DM[keep, :]
        data = data[:, keep]

    # ── Drop zero columns ────────────────────────────────────────────────────
    col_norms = np.abs(DM).sum(axis=0)
    is_zero   = col_norms == 0
    fix_idx   = np.where(is_zero)[0]         # original positions of zero cols
    FixDM     = is_zero.any()
    if FixDM:
        DM = DM[:, ~is_zero]

    # ── GLM: beta map and residuals ──────────────────────────────────────────
    # MATLAB: b = (pinv(DM'*DM) * DM' * data')'
    #         e = (data' - DM * b')'
    DtD_inv = pinv(DM.T @ DM)
    b       = (DtD_inv @ DM.T @ data.T).T          # (voxels, regressors)
    e       = (data.T - DM @ b.T).T                # (voxels, time)

    var_data = data.var(axis=1, ddof=0) * data.shape[1]  # matches MATLAB var(...,1)*N
    r_sqrd   = 1.0 - (e ** 2).sum(axis=1) / var_data

    # ── Re-insert zero-columns into beta map ─────────────────────────────────
    if FixDM:
        for idx in fix_idx:
            idx = int(idx)
            b = np.concatenate(
                [b[:, :idx], np.zeros((b.shape[0], 1)), b[:, idx:]],
                axis=1
            )

    return b, e, DM, EDM, keep, r_sqrd


def normalize_rows(data):
    """
    Normalize each row to zero mean and unit L2 norm (mimics MATLAB's normr).
    This enables fast Pearson correlation via dot products.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return data / norms


def normcND(data):
    """ 
    NORMCND returns a column-normed matrix. It is assumed that the matrix is 2D.
    """
    vecnorm = lna.norm(data, ord = 2, axis = 0)
    data = data / vecnorm

    return data


def normrND(data):
    """
    NORMRND returns a row-normed matrix. It is assumed that the matrix is 2D. Updated for broader compatability.
    """
    dataNorm = np.sqrt(np.sum(data**2))
    data = data / dataNorm
    data[np.argwhere(np.isfinite(data) == False)] = 0

    return data


def parcel_based_fc(data, parcels, dim, params=None):
    """
    Calculate parcel-based functional connectivity between a volumetric dataset
    and a set of volumetrically defined parcels.

    Parameters
    ----------
    data : np.ndarray
        Volumetric data. Time is assumed to be the last dimension.
        Expected to already be preprocessed (demeaned + normalized).
        If not 2D, data is reshaped to (n_voxels, n_timepoints).
    parcels : np.ndarray
        Volume of integers defining parcel locations (0 = background).
        Must be in the same space as `data`.
    dim : tuple or array-like
        Spatial dimensions of the data volume (e.g. (x, y, z)).
    params : dict, optional
        Reserved for future use (e.g. custom masks).

    Returns
    -------
    fc_maps : np.ndarray, shape (x, y, z, n_parcels)
        Fisher-Z transformed seed correlation maps for each parcel.
    fc_matrix : np.ndarray, shape (n_parcels, n_parcels)
        Fisher-Z transformed parcel-to-parcel correlation matrix.
    parcel_tt : np.ndarray, shape (n_timepoints, n_parcels)
        Mean ROI timecourse for each parcel.
        n_timepoints matches the time dimension of the input data
        (i.e. the number of kept frames after any temporal masking applied
        upstream, NOT the full original recording length).

    Notes
    -----
    The correlation maps are computed in a single matrix multiply rather than
    a per-parcel loop, making this significantly faster for large parcel sets.
    Both `data` and the parcel seed timecourses are row-normalized before the
    dot product so the result is equivalent to Pearson r.
    """
    if params is None:
        params = {}
    # --- Shape handling ---
    dims = data.shape
    n_t = dims[-1]  # time is always the last dimension

    if data.ndim > 2:
        data = np.moveaxis(data, 0, -1)  
        n_t = data.shape[-1]
        data = data.reshape(-1, n_t)    

    # --- Parcel setup ---
    parcels_flat = parcels.ravel()
    u_parcels = np.unique(parcels_flat[parcels_flat != 0])
    n_parcels = len(u_parcels)

    # --- Build parcel mean timecourses ---
    parcel_tt = np.zeros((n_t, n_parcels))

    if data.ndim > 2:
        data = np.moveaxis(data, 0, -1)  
        n_t = data.shape[-1]
        spatial_dims = data.shape[:3]   
        data = data.reshape(-1, n_t)
    else:
        n_t = data.shape[-1]
        spatial_dims = tuple(dims[:3])
        
    print(f"Calculating correlations for {n_parcels} parcels...")
    
    for k, label in enumerate(u_parcels):
        vol_mask = (parcels_flat == label)   # boolean, shape (n_voxels,)
        n_s = vol_mask.sum()
        if n_s == 0:
            continue
        parcel_tt[:, k] = data[vol_mask, :].sum(axis=0) / n_s  # mean over parcel voxels

    # --- Vectorized correlation: all parcels in one matrix multiply ---

    # Build parcel mean timecourses from already-normalized data
    data_norm = ndot.normalize_rows(data)   # normalize data rows first

    parcel_tt = np.zeros((n_t, n_parcels))
    for k, label in enumerate(u_parcels):
        vol_mask = (parcels_flat == label)
        n_s = vol_mask.sum()
        if n_s == 0:
            continue
        parcel_tt[:, k] = data_norm[vol_mask, :].sum(axis=0) / n_s  # mean of normalized rows

    # Dot product directly — do NOT re-normalize parcel_tt
    r_all = data_norm @ parcel_tt     # (n_voxels, n_parcels)
    fc_maps = ndot.fisher_r2z(r_all)

    # --- Reshape fc_maps to volume ---
    spatial_dims = parcels.shape

    fc_maps = fc_maps.reshape(*spatial_dims, n_parcels)

    # --- Parcel-to-parcel FC matrix ---
    fc_matrix = ndot.fisher_r2z(np.corrcoef(parcel_tt.T))

    return fc_maps, fc_matrix, parcel_tt

