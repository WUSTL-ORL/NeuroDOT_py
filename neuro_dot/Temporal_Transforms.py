# General imports
import math
import fractions
import numpy as np
import numpy.linalg as lna
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import scipy.interpolate
import scipy as scp 
import numpy.matlib as mlb
import numpy.matlib as nm
import functools as ft
import scipy.signal as sig
import sympy as sym
import copy


from math import trunc
from pickle import NONE
from numpy import float64, matrix
from numpy.lib.shape_base import expand_dims
from matplotlib.pyplot import colorbar, colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

from Analysis import anlys


class tx4m:
    

    def detrend_tts(data_in):
        """
        DETREND_TTS Performs linear detrending.

        data_out = DETREND_TTS(data_in) takes a raw light-level data array
        "data_in" of the format MEAS x TIME and removes the straight-line fit
        along the TIME dimension from each measurement, returning it as
        "data_out".

        See Also: LOGMEAN.
        """
        ## Parameters and initialization
        dims = np.shape(data_in)
        Nt = dims[-1]
        NDtf = np.ndim(data_in) > 2

        ## N-D Input.
        if NDtf:
            data_in = np.reshape(data_in, len(data_in)/Nt, Nt)

        ## Detrend.
        data_out = np.squeeze(sig.detrend(data_in[:,None])) # transposing isn't necessary for this process in Python as it is in Matlab

        ## Remove mean
        if data_out.ndim == 1: #for when data_in is a vector
            meanRow = np.ndarray.mean(data_out, dtype = np.float64)
            data_out= data_out - meanRow
        else:
            meanRow = np.ndarray.mean(data_out, axis = 1,dtype = np.float64)
            data_out= data_out - meanRow[:,None]

        ## N-D Output.
        if NDtf:
            data_out = np.reshape(data_out, dims)
        
        return data_out


    def nextpow2(N):
        """ Function for finding the next power of 2 """
        n = 1
        while n < N: n = n^2
        return n


    def fft_tts(data, framerate):
        """
        FFT_TTS Performs an FFT.
        ftmag = FFT_TTS(data, framerate) takes a data array "data" of
        the MEAS x TIME format, pads it timewise to the next highest power of
        two (for better performance), performs the fast Fourier transform of
        each channel using the built-in MATLAB function FFT, normalizes by the
        padded time length, and takes the first half of the transformed data
        (which is the positive half of the frequency domain). The result is
        output into "ftmag".

        [ftmag, ftdomain] = FFT_TTS(data, framerate) also returns the
        corresponding normalized frequency domain "ftdomain", which extends
        from 0 to the Nyquist frequency, which is calculated from the input
        "framerate".
    
        [ftmag, ftdomain, ftpower] = FFT_TTS(data, framerate) also returns the
        power spectrum, which is the absolute value of the magnitude, squared.
    
        [ftmag, ftdomain, ftpower, ftphase] = FFT_TTS(data, framerate) also
        returns the phase at each frequency, as calculated by the MATLAB
        ANGLE function.
    
        Dependencies: NORMALIZE2RANGE_TTS.
        
        See Also: LOGMEAN, FFT, POW2, NEXTPOW2, ANGLE.
        """
        ## Parameters and Initialization.
        dims = data.shape
        Nt = dims[-1]
        NDtf = (len(dims) > 2)
        Ndft = 2 ** math.ceil(math.log2(abs(Nt))); # Zero pack to a power of 2.
        Nf = int(1 + Ndft / 2)
        
        # %% N-D Input.
        if NDtf:
            data = np.reshape(data, len(data)/Nt, Nt)

        # %% Remove mean.
        meanRow = np.ndarray.mean(data, axis = 1,dtype = np.float64)
        normdata = data - meanRow[:,None]

        # %% Prep data for FFT. 
        ftdomain = (framerate / 2) * np.linspace(0, 1, Nf) # domain of FFT: [zero:Nyquist]

        # %% Perform FFT.
        P = scp.fft.fft(normdata, Ndft, 1) / Ndft # Do FFT in TIME dimension and normalize by Ndft
        ftmag = math.sqrt(2) * abs(P[:, 0:Nf]) # Take positive frequencies, x2 for negative frequencies.

        # %% N-D Output.
        if NDtf:
            ftmag = np.reshape(ftmag, data.shape[0], Nf)

        # %% Other outputs.
        ftpower = abs(ftmag) ** 2
        ftphase = np.angle(P[:, 1:Nf])

        return ftdomain, ftmag, ftpower, ftphase



    def gethem(data, info, sel_type  = 'r2d', value = [10,16]):
        """
        GETHEM Calculates mean across set of measurements.
        
        hem = GETHEM(data, info) takes a light-level array "data" of the format
        MEAS x TIME, and using the scan metadata in "info.pairs" averages the
        measurements for each wavelength present. The result is
        referred to as the "hem" of a measurement set. If there is a
        good measurements logical vector present in "info.MEAS.GI", it will be
        applied to the data; otherwise, "info.MEAS.GI" will be set to true for
        all measurements (i.e., all measurements are assumed to be good). The
        variable "hem" is output in the format WL x TIME.
        
        hem = GETHEM(data, info, sel_type, value) allows the user to set the
        criteria for determining shallow measurements. "sel_type" can be 'r2d',
        'r3d', or 'NN', corresponding to the columns of the "info.pairs" table,
        and "value" can either take the form of a two-element "[min, max]"
        vector (for 'r2d' and 'r3d'), or a scalar or vector containing all
        nearest neighbor numbers to be averaged. By default, this function
        averages the first nearest neighbor.
        
        See Also: REGCORR, DETREND_TTS.
        """

        ## Parameters and Initialization
        Nm = np.shape(data)[0]
        Nt = np.shape(data)[1]
        cs = np.unique(info['pairs']['WL'])
        Nc = len(cs)
        hem = np.zeros(shape = [Nc,Nt])

        if sel_type == 'r2d':
            keep_R_NN = np.logical_and((info['pairs']['r2d'] >= value[0]), info['pairs']['r2d'] <= value[1]).astype(np.uint8)
        elif sel_type == 'r3d':
            keep_R_NN = np.logical_and((info['pairs']['r3d'] >= value[0]), info['pairs']['r3d'] <= value[1]).astype(np.uint8)
        elif sel_type == 'NN':
            keep_R_NN = np.zeros(shape = info['pairs']['NN'])
            k = value
            while k == value:
                keep_R_NN = np.logical_or(keep_R_NN, (info['pairs']['NN'] == k)).astype(np.uint8)
                k = k+1

        if np.logical_and(('MEAS' in info), (not 'GI' in info['MEAS'])):
            info['MEAS']['GI'] = np.ones(shape = (Nm, 1), dtype = np.bool8)
        elif not 'MEAS' in info:
            info['MEAS'] = { 'GI': np.ones(shape = (Nm, 1), dtype = np.bool8)}

        k = 0
        while k <=1:
            keep = np.logical_and(np.logical_and(keep_R_NN, (info['pairs']['WL'] == cs[k])), info['MEAS']['GI']).astype(np.uint8)
            hem[k, :] = np.mean(data[np.squeeze(np.argwhere(keep == 1)), :], 0) #np.mean(data[keep, :], 0)
            k = k + 1
        return hem


    def highpass(data_in, omegaHz, frate, params = None):
        """ 
        HIGHPASS Applies a zero-phase digital highpass filter.

            data_out = HIGHPASS(data_in, omegaHz, frate) takes a light-level array
            "data_in" in the MEAS x TIME format and applies to it a
            forward-backward zero-phase digital highpass filter at a Nyquist cutoff
            frequency of "omegaHz * (2 * frate)", returning it as "data_out".

            This function also removes the linear component of the input data.

        See Also: LOWPASS, LOGMEAN, FILTFILT.
        """
        ## Parameters and Initialization.
        if params == None:
            params = {}

        if 'poles' in params:
            poles = params['poles']
        else:
            poles = 5
        if 'pad' in params:
            Npad = int(params['pad'])
        else:
            Npad = 100
        if 'detrend' in params:
            DoDetrend = params['detrend']
        else:
            DoDetrend = 1
        if 'DoPad' in params:
            DoPad = params['DoPad']
        else:
            DoPad = 1

        if not DoPad:
            Npad = 0

        if isinstance(data_in, np.single):
            isSing = 1
        else:
            isSing = 0

        dims = np.shape(data_in)
        Nt = dims[-1]  # Assumes time is always the last dimension.
        NDtf = np.ndim(data_in) > 2

        ## N-D Input.
        if NDtf:
            data_in = np.reshape(data_in, len(data_in)/Nt, Nt)
        
        Nm=np.shape(data_in)[0]

        ## Calculate Nyquist frequency and build filter.
        omegaNy = omegaHz * (2 / frate)
        [b, a] = sig.butter(poles, omegaNy, 'highpass')
        
        ## Remove mean
        if data_in.ndim == 1: #for when data_in is a vector
            meanRow = np.ndarray.mean(data_in, dtype = np.float64)
            data_in= data_in - meanRow
        else:
            meanRow = np.ndarray.mean(data_in, axis = 1,dtype = np.float64)
            data_in= data_in - meanRow[:,None]

        ## Detrend.
        if DoDetrend:
            data_in = np.squeeze(sig.detrend(data_in[:,None])) # transposing isn't necessary for this process in Python as it is in Matlab

        ## Zero pad    
        array = np.zeros((int(Nm),int(Npad)))
        data_in = np.append(array,data_in,axis = 1)
        data_in = np.append(data_in,array,axis = 1)

        ## Forward-backward filter data for each measurement.
        if isSing:
            data_in = np.double(data_in)
        data_out = sig.filtfilt(b, a, data_in, padlen = 3*(max(len(b), len(a)) - 1))

        if isSing:
            data_out = np.single(data_out)
        data_out = data_out[:,(Npad):(-Npad)] 

        ## Remove mean
        meanRow = np.ndarray.mean(data_out, axis = 1,dtype = np.float64)
        data_out = data_out - meanRow[:,None]
        ## N-D Output.
        if NDtf:
            data_out = np.reshape(data_out, dims)
        
        #print(data_out)
        return data_out


    def logmean(data_in):
        """
        LOGMEAN Takes the log-ratio of raw intensity data.
        
        data_out = LOGMEAN(data_in) takes a light-level data array "data_in" of
        the format MEAS x TIME, and takes the negative log of each element of a
        row divided by that row's average. The result is output into "data_out"
        in the same MEAS x TIME format.
        
        The formal equation for the LOGMEAN operation is:
            Y_{out} = -log(phi_{in} / <phi_{in}>)
        
        If the raw optical data phi is complex (as in the frequency domain 
        case), Y behaves a bit differently. Phi can be defined in terms of Real
        and Imaginary parts: Phi = Re(Phi) + 1i.*Im(Phi), or in terms of it's
        magnitude (A) and phase (theta): Phi = A.*exp(1i.*theta).
        The temporal average of Phi (what we use for baseline) is best
        calculated on the Real/Imaginary decription: 
        
            Phi_o=<phi>=mean(data_in,2) = A_o*exp(i*(th_o));
        
        Taking the logarithm of complex ratio:
        
            Y_Rytov=-log(phi/<phi>)=-log[A*exp(i*th)/A_o*exp(i*th_o)]
                                    =-[log(A/A_o) + i(th-th_o)];
        
            Y_Rytov_Re=-log(abs(data_in/Phi_o));
            Y_Rytov_Im=-angle(data_in/Phi_o);
        
        Though this looks like 1 complex number, these components of Y should
        not mix, so the imaginary component will be tacked onto the end of the
        measurement list to keep them separate.
        
        
        Example: If data = [1, 10, 100; exp(1), 10*exp(1), 100*exp(1)];
        
        then LOGMEAN(data) yields [3.6109, 1.3083, -.9943; 3.6109, 1.3083,
        -.9943].
        
        See Also: LOWPASS, HIGHPASS.
        """

        # Parameters and Initialization.
        dims = np.array(np.shape(data_in))
        Nt = dims[-1] # Assumes time is always the last dimension, -1 is the index of the last dimension of an array in Python
        NDtf = np.ndim(data_in) > 2
        isZ = not np.any(np.isreal(data_in)) #check if there is at least one element in data_in that is a complex number
        # N-D Input.
        if NDtf:
            data_in = np.reshape(data_in, [], Nt)

        # Perform Logmean.
        Phi_0 = np.mean(data_in,1)
        X = data_in / Phi_0[:,None]

        if not isZ:
        #   data_out = {}
            data_out = np.double(-np.log(X))
        else:
            Y_Rytov_Re = -np.log(abs(X))
            Y_Rytov_Im = -np.angle(X)

            data_out = np.concatenate((Y_Rytov_Re, Y_Rytov_Im), axis = 0) #concatenate along first dimension
            dims[0] = 2*dims[0]

        # Fix any NaNs.
        data_out[not data_in[:].any()]=0

        # N-D Output.
        if NDtf:
            data_out = np.reshape(data_out, dims)

        # Return output
        return (data_out, Phi_0)    

    
    def lowpass(data_in, omegaHz,frate,params = None):
        """
        LOWPASS Applies a zero-phase digital lowpass filter.
        
        data_out = LOWPASS(data_in, omegaHz, frate) takes a light-level array
        "data_in" in the MEAS x TIME format and applies to it a
        forward-backward zero-phase digital lowpass filter at a Nyquist cutoff
        frequency of "omegaHz * (2 * frate)", returning it as "data_out".
        
        This function also removes the linear component of the input data.
        
        See Also: HIGHPASS, LOGMEAN, FILTFILT.
        """
        ## Parameters and Initialization.
        if params == None:
            params = {}

        if 'poles' in params:
            poles = params['poles']
        else:
            poles = 5
        if 'pad' in params:
            Npad = int(params['pad'])
        else:
            Npad = 100
        if 'detrend' in params:
            DoDetrend = params['detrend']
        else:
            DoDetrend = 1
        if 'DoPad' in params:
            DoPad = params['DoPad']
        else:
            DoPad = 1

        if not DoPad:
            Npad = 0

        if isinstance(data_in, np.single):
            isSing = 1
        else:
            isSing = 0

        dims = np.shape(data_in)
        Nt = dims[-1]  # Assumes time is always the last dimension.
        NDtf = np.ndim(data_in) > 2

        ## N-D Input.
        if NDtf:
            data_in = np.reshape(data_in, len(data_in)/Nt, Nt)

        Nm=np.shape(data_in)[0]

        ## Calculate Nyquist frequency and build filter.
        omegaNy = omegaHz * (2 / frate)
        [b, a] = sig.butter(poles, omegaNy, 'lowpass')
        
        ## Remove mean
        if data_in.ndim == 1: #for when data_in is a vector
            meanRow = np.ndarray.mean(data_in, dtype = np.float64)
            data_in= data_in - meanRow
        else:
            meanRow = np.ndarray.mean(data_in, axis = 1,dtype = np.float64)
            data_in= data_in - meanRow[:,None] 

        ## Detrend.
        if DoDetrend:
            data_in = np.squeeze(sig.detrend(data_in[:,None], type = 'linear')) # transposing isn't necessary for this process in Python as it is in Matlab


        ## Zero pad    
        array = np.zeros((int(Nm),int(Npad)))
        data_in = np.append(array,data_in,axis = 1)
        data_in = np.append(data_in,array,axis = 1)

        ## Forward-backward filter data for each measurement.
        if isSing:
            data_in = np.double(data_in)
        data_out = sig.filtfilt(b, a, data_in, padlen = 3*(max(len(b), len(a)) - 1))

        if isSing:
            data_out = np.single(data_out)
        data_out = data_out[:,(Npad):(-Npad)] 

        ## Detrend.
        if DoDetrend:
            data_out = np.squeeze(sig.detrend(data_out[:,None], type = 'linear')) # transposing isn't necessary for this process in Python as it is in Matlab

        ## Remove mean
        if data_in.ndim == 1: #for when data_in is a vector
            meanRow = np.ndarray.mean(data_out, dtype = np.float64)
            data_out= data_out - meanRow
        else:
            meanRow = np.ndarray.mean(data_out, axis = 1,dtype = np.float64)
            data_out = data_out - meanRow[:,None]
        
        ## N-D Output.
        if NDtf:
            data_out = np.reshape(data_out, dims)
        

        return data_out


    def regcorr(data_in, info, hem):
        
        """
        REGCORR Performs regression correction by wavelengths.

        [data_out, R] = REGCORR(data_in, info, hem) takes a light-level data
        array "data_in" of the format MEAS x TIME, and using the scan metadata
        in "info.pairs" and a WL x MEAS "hem" array generated by GETHEM,
        performs a regression correction for each wavelength of the data, which
        is returned in the MEAS x TIME array "data_out". The corresponding
        correlation coefficients for each measurement are returned in "R" as a
        MEAS x 1 array.

        If y_{r} is the signal to be regressed out and y_{in} is a data
        time trace (either source-detector or imaged), then the output
        is the least-squares regression: 
        y_{out} = y_{in} - y_{in}(<y_{in},y_{r}>/|y_{r}|^2). 
        Additionally, the correlation coefficient is given by: 
        R=(<y_{in},y_{r}>/(|y_{in}|*|y_{r}|)).

        See Also: GETHEM, DETREND_TTS.
        """
        ## Parameters and Initialization.
        Nm = np.shape(data_in)[0]
        Nt = np.shape(data_in)[1]
        cs = np.unique(info['pairs']['WL']) #WLs
        Nc = len(cs) #Number of WLs
        data_out = np.zeros(shape = (Nm, Nt))
        R = np.zeros(shape = (Nm, 1))

        ## Regression correction.
        for k in range(0, Nc):
            keep = info['pairs']['WL'] == cs[k]
            temp = np.transpose(data_in[keep, :])

            g = np.transpose(hem[k, :]) #regressor/noise signal in correct orientation
            g.shape = (g.shape[0], 1)
            gp = lna.pinv(g)
            beta = gp.dot(temp)
            data_out[keep, :] = np.transpose(temp - g.dot(beta)) #linear regression

            R[keep] = np.transpose(anlys.normrND(np.transpose(g)).dot(anlys.normcND(temp))) #correlation coefficient
        
        return data_out, R


    def resample_tts(data_in, info_in, omega_resample = 1, tol = 0.001, framerate = 0): # returns data_out and info_out
        """
        RESAMPLE_TTS Resample data while maintaining linear signal component.

        [data_out, info_out] = RESAMPLE_TTS(data_in, info_in, tHz, tol,
        framerate) takes a raw light-level data array "data_in" of the format
        MEAS x TIME, and resamples it (typically downward) to a new frequency
        using the built-in MATLAB function RESAMPLE. The new sampling frequency
        is calculated as the ratio of input "omega_resample" divided by
        "framerate" (both scalars), to within the tolerance specified by "tol".

        This function is needed because the linear signal components, which can
        be important in other NeuroDOT pipeline calculations, can be
        inadvertently removed by downsampling using RESAMPLE alone.

        Note: This function resamples synch points in addition to data. Be sure
        to take care that your data and synch points match after running this
        function! "info.paradigm.init_synchpts" stores the original synch
        points if you need to restore them.

        See Also: DETREND_TTS, RESAMPLE.
        """
        ## Parameters and Initialization.
        info_out = copy.deepcopy(info_in)
        dims = np.shape(data_in)
        Nt = dims[-1]
        NDtf = np.ndim(data_in) > 2
        if framerate == 0:
            if np.logical_and(np.logical_and('system' in info_in, info_in['system'] != {}), np.logical_and('framerate' in info_in['system'], info_in['system']['framerate'] != {})):
                framerate = info_in['system']['framerate'] 

        ## N-D Input.
        if NDtf:
            data_in = np.reshape(data_in, [], Nt)


        ## Approximate desired resampling ratio as a fraction.
        fract = sym.nsimplify((omega_resample/framerate), tolerance = tol) #in matlab tolerance is set to 1e-5, in order to reproduce outputs from neuroDOT matlab, tolerance is set to 0.001 in neuroDOT
        N = fract.numerator
        D = fract.denominator
        info_out['system']['framerate'] = omega_resample


        ## Remove linear fit.
        Nt = np.shape(data_in)[1]
        d0 = data_in[:, 0] # start point
        dF = data_in[:, Nt-1] # end point
        beta = -d0

        alpha1 = np.divide((d0-dF), (Nt-1)) # slope for linear fit
        cols = np.arange(0, Nt, 1)  # bsxfun multiplication
        alpha_full = np.ones((dims[0], Nt))
        for x in cols:
            alpha_full[:, x] = cols[x] * alpha1
        beta.shape = (np.shape(beta)[0],1)
        correction = alpha_full + beta # bsxfun addition
        corrsig = data_in + correction


        ## Resample with endpoints pinned to zero.
        rawresamp = np.transpose(sig.resample_poly(np.transpose(corrsig[:]), N, D)) #using scipy signal polyphase resampling


        ## Add linear fit back to resampled data.
        alpha2 = alpha1 * (D/N) 
        Nt = np.shape(rawresamp)[1] # column dimension
        cols = np.arange(0, Nt, 1) # bsxfun multiplication
        alpha_full = np.ones((dims[0], Nt))
        for x in cols:
            alpha_full[:, x] = cols[x] * alpha2
        correction = alpha_full + beta #bsxfun addition
        data_out = rawresamp - correction


        ## Fix synch pts to new framerate.
        if 'paradigm' in info_in:
            if 'init_synchpts' in info_in['paradigm']:
                info_out['paradigm']['synchpts'] = np.round(np.divide(np.dot(N, info_out['paradigm']['init_synchpts']), D)) # info_out.paradigm.synchpts = round(N .* info_out.paradigm.init_synchpts ./ D);
                info_out['paradigm']['synchpts'][np.argwhere(info_out['paradigm']['synchpts'] == 0)] = 1
            elif 'synchpts' in info_in['paradigm']:
                info_out['paradigm']['init_synchpts'] = info_out['paradigm']['synchpts']
                info_out['paradigm']['synchpts'] = np.round(np.divide(np.dot(N, info_out['paradigm']['synchpts']), D))
                info_out['paradigm']['synchpts'][np.argwhere(info_out['paradigm']['synchpts'] == 0)] = 1


        ## N-D Output.
        if NDtf:
            data_out = np.reshape(data_out, (dims[0:-1], Nt))

        return data_out, info_out

