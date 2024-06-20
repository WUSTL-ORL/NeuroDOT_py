# General imports
import numpy as np
import scipy as scp 
import scipy.ndimage as ndi
import numpy.linalg as lna



def reconstruct_img(lmdata, iA):
    '''
    RECONSTRUCT_IMG Performs image reconstruction by wavelength using the inverted A-matrix.
    
    img = RECONSTRUCT_IMG(data, iA) takes the inverted VOX x MEAS
    sensitivity matrix "iA" and right-multiplies it by the logmeaned MEAS x
    TIME light-level matrix "lmdata" to reconstruct an image in voxel
    space. 
    
    The image is output in a VOX x TIME matrix "img".
    
    See Also: TIKHONOV_INVERT_AMAT, SMOOTH_AMAT, SPECTROSCOPY_IMG,
    FINDGOODMEAS.
    '''
    ## Parameters and Initialization
    units_scaling = 1/100 # Assuming OptProp in mm^-1 
    
    ## Reconstruct.
    cortex_mu_a = iA @ lmdata

    ## Correct units and convert to single.
    cortex_mu_a = np.single(np.multiply(cortex_mu_a, units_scaling))

    return cortex_mu_a

def smooth_Amat(iA_in, dim, gsigma):
    '''
    SMOOTH_AMAT Performs Gaussian smoothing on a sensitivity matrix.

    iA_out = SMOOTH_AMAT(iA_in, dim, gsigma) takes the inverted VOX x MEAS
    sensitivity matrix "iA_in" and performs Gaussian smoothing in the 3D
    voxel space on each of the concatenated measurement matrices within,
    returning it as "iA_out". The user specifies the Gaussian filter
    half-width "gsigma".

    See Also: TIKHONOV_INVERT_AMAT, RECONSTRUCT_IMG, FINDGOODMEAS.
    '''
    # Parameters and Initialization.
    Nvox = np.shape(iA_in)[0]
    Nm = np.shape(iA_in)[1]
    iA_out = np.zeros((Nvox, Nm))

    nVx =int(dim['nVx'])
    nVy = int(dim['nVy'])
    nVz = int(dim['nVz'])

    gsigma = gsigma / dim['sV']

    # Preallocate voxel space.
    if 'Good_Vox' in dim:
        GV = dim['Good_Vox'].astype(int)
    else:
        GV = np.ones((nVx, nVy, nVz)).astype(int) # WARNING: THIS RUNS WAY SLOWER.

    # Do smoothing in parallel.
    for k in range(0, Nm):
        iAvox = np.zeros((nVx, nVy, nVz)) # Set up temp iAvox
        iAvox = np.reshape(iAvox, (nVx * nVy * nVz, ), order = 'F')     # iAvox needs to be a vector for GV to be used as an index
        iAvox[GV-1] = iA_in[:, k]                                       
        iAvox = np.reshape(iAvox, (nVx, nVy, nVz), order = 'F')
        # Gaussian smoothing to replicate Matlab's imgaussfilt3()
        # imgaussfilt3's default filter size is 2*ceil(2*SIGMA)+1 so we set the truncate parameter to ceil(2*gsigma)/gsigma (=2, for input gsigma of 3)
        # imgaussfilt3 default pad info: "Input image values outside the bounds of the image are assumed equal to the nearest image border value." so we set mode = 'nearest'
        iAvox = ndi.gaussian_filter(iAvox, gsigma, mode = 'nearest', truncate=np.ceil(2*gsigma)/gsigma)
        iAvox= np.reshape(iAvox, (nVx * nVy * nVz, ), order = 'F')      # reshape iAvox like above or else GV cannot be used as index
        iA_out[:,k] = np.single(iAvox[GV-1])                            
        iAvox = np.reshape(iAvox, (nVx, nVy, nVz), order = 'F')
    
    return iA_out

def spectroscopy_img(cortex_mu_a, E):
    '''
    SPECTROSCOPY_IMG Completes the Beer-Lambert law from a reconstructed image.

    img_out = SPECTROSCOPY_IMG(img_in, E) takes the reconstructed VOX x
    TIME x WL mua image "img_in" and multiplies it by the inverse
    of the extinction coefficient matrix "E" to create an output VOX x TIME
    x HB matrix "img_out", where HB 1 and 2 are the voxel-space time series
    images for HbO and HbR, respectively.

    See Also: RECONSTRUCT_IMG, AFFINE3D_IMG.
    '''
    
    ## Parameters and Initialization.
    Nvox = cortex_mu_a.shape[0]
    Nt = cortex_mu_a.shape[1]
    Nc = cortex_mu_a.shape[2]
    E1 = E.shape[0]
    E2 = E.shape[1]
    umol_scale = 1000

    ## Check compatibility of the image and E matrix.
    if E1 != Nc:
        raise ValueError('Error: The image wavelengths and spectroscopy matrix dimensions do not match.')

    ## Invert Spectroscopy Matrix.
    iE = np.linalg.inv(E)

    #Initialize Outputs
    cortex_hb = np.zeros((Nvox, Nt, Nc), dtype = np.float64)
    for k in range(0, E2):
        temp = np.zeros((Nvox, Nt))
        for l in range(0, E1):
            temp = temp + np.multiply(np.squeeze(iE[k, l]), np.squeeze(cortex_mu_a[:,:,l]))
        cortex_hb[:, :, k] = temp

    cortex_hb = np.multiply(cortex_hb, umol_scale) # Fix units to umol

    return cortex_hb

def Tikhonov_invert_Amat(A, lambda1, lambda2):
    '''
    TIKHONOV_INVERT_AMAT Inverts a sensitivity "A" matrix.

    iA = TIKHONOV_INVERT_AMAT(A, lambda1, lambda2) allows the user to
    specify the values of the "lambda1" and "lambda2" parameters in the
    inversion calculation. lambda2 for spatially-variant regularization,
    is optional.

    See Also: SMOOTH_AMAT, RECONSTRUCT_IMG, FINDGOODMEAS.
    ''' 
    if not np.isreal(A.any):  # If complex A, sep into [Re;Im] first
        A = np.concatenate((A.re, A.imag), axis = 0)
    Nm = np.shape(A)[0]
    Nvox = np.shape(A)[1]
    if lambda2:
        svr = 1
    else:
        svr = 0

    ## Spatially variant regularization
    if svr:
        ll_0 = np.sum((A**2), axis = 0) 
        ll = np.sqrt(ll_0+lambda2*max(ll_0)) # Adjust with Lambda2 cut-off value
        A = A/ll

    ## Take the pseudo-inverse.
    if Nvox < Nm:
        Att = np.zeros(Nvox, dtype = np.single)
        Att = np.single(A.astype(np.transpose(A).astype(np.float64) @ np.float64))
        ss = lna.norm(Att, ord = 2) # numpy.linalg.norm() with ord = 2 is the matrix 2-norm 
        penalty = np.multiply(np.sqrt(ss), lambda1)
        iA = lna.lstsq(Att + np.multiply(penalty**2, np.eye(Nvox, dtype = np.uint8)), np.transpose(A))
    else:
        Att = np.zeros(Nm, dtype = np.single)
        Att = np.single(A.astype(np.float64) @ np.transpose(A).astype(np.float64))  
        ss = lna.norm(Att)
        penalty = np.multiply(np.sqrt(ss), lambda1)
        # In matlab, the following two lines are written as: iA = A' / (Att + penalty .^ 2 .* eye(Nm, 'single'));
        # In python, it is impossible to divide a (m,n) matrix by (n,n), when m or n != 1
        # Instead, we multiply A' by the inverse of the divisor 
        iAtt = lna.inv((Att + np.multiply(penalty**2, np.eye(Nm, dtype = np.uint8))))
        iA = np.matmul(np.transpose(A),iAtt)

    ## Undo spatially variant regularization
    if svr:
        ll.shape = (1, ll.shape[0]) # add singleton dimension to ll
        iA = iA / np.transpose(ll)

    return iA