# General imports
import numpy as np
import numpy.linalg as lna

from numpy.lib.shape_base import expand_dims
from Matlab_Equivalent_Functions import matlab






class anlys:

   
    def BlockAverage(data_in, pulse, dt, Tkeep = 0):
        """ 
        BLOCKAVERAGE Averages data by stimulus blocks.
        data_out = BLOCKAVERAGE(data_in, pulse, dt) takes a data array
        "data_in" and uses the pulse and dt information to cut that data 
        timewise into blocks of equal length (dt), which are then averaged 
        together and output as "data_out".
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
        data_in[:, np.argwhere(Tkeep == False)] = np.NaN

        ## Cut data into blocks.
        Nm = np.shape(data_in)[0]
        blocks = np.zeros((Nm, dt, Nbl))

        for k in range(0, Nbl):
            pulse_k = int(pulse[k])
            if (pulse[k] + dt -1) <= Nt:
                blocks[:, :, k] = data_in[:, pulse_k-1:pulse_k + dt-1] #Need to subtract 1 from both indices to account for 0 indexing in python
            else:
                dtb = (pulse_k-1) + dt - 1 - Nt
                nans = np.empty(shape = (np.shape(data_in)[0], dtb)) #need multiple lines to create an array of nans in python
                nans[:] = np.NaN
                blocks[:, :, k] = np.concatenate((data_in[:, (pulse_k-1):Nt], nans), axis = 1) #need to subtract 1 from start index: pulse[k] due to zero indexing, also need to use Nt as final index to get correct size

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
            newshape_blocks = tuple(np.append(np.array(dims[0:-1]), (dt, Nbl))) #create tuple for deisred output shape for blocks, different from previous newshape bc Nbl is also appended
            blocks = np.reshape(blocks, newshape_blocks)


        return BA_out, BSTD_out, BT_out, blocks


    def CalcGVTD(data):
        """
        This function calculates the Root Mean Square across measurements (be
        they log-mean light levels or voxels) of the temporal derivative. The
        data is assumed to have measurements in the first and time in the last 
        dimension. Any selection of measurement type or voxel index must be done
        outside of this function.
        """
        # Double check data has correct dimensions
        # Dsizes=size(data);
        Dsizes = np.shape(data)
        Ndim = len(Dsizes)
        if Ndim > 2:
            data = np.reshape(data, [], Dsizes[-1])
        
        # 1st Temporal Derivative
        Ddata = data - np.roll(data, np.array([0, -1]), np.array([0,-1])) #if you are shifting a matrix, and shifting both axes by a value, you MUST specify both axes as the third argument in roll()

        # RMS across measurements
        GVTD = np.concatenate(([0], matlab.rms_py(Ddata[:,0:-1])), axis = 0)

        return GVTD

    
    def FindGoodMeas(data, info_in, bthresh = 0.075):
        from Matlab_Equivalent_Functions import matlab
        from Temporal_Transforms import tx4m

        """ 
        FINDGOODMEAS Performs "Good Measurements" analysis.
        info_out = FINDGOODMEAS(data, info_in) takes a light-level array "data"
        in the MEAS x TIME format, and calculates the std of each channel
        as its noise level. These are then thresholded by the default value of
        0.075 to create a logical array, and both are returned as MEAS x 1
        columns of the "info.MEAS" table. If pulse synch point information
        exists in "info.system.synchpts", then FINDGOODMEAS will crop the data
        to the start and stop pulses.
        info_out = FINDGOODMEAS(data, info_in, bthresh) allows the user to
        specify a threshold value.
        See Also: PLOTCAPGOODMEAS, PLOTHISTOGRAMSTD.
        """
        # Parameters and Initialization.
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
        info_out = info_in.copy() #create info_out
        try:
            GVwin
        except NameError:
            GVwin = 600
        if not 'paradigm' in info_out:
            info_out['paradigm'] = {}
        if not bthresh in locals():
            bthresh = 0.075 # Empirically derived threshold value.
        dims = np.shape(data)
        Nt = dims[-1] # Assumes time is always the last dimension, -1 is the index of the last dimension of an array in Python
        NDtf = np.ndim(data) > 2
        if GVwin > (Nt-1):
            GVwin = (Nt-1)
        
        # N-D Input.
        if NDtf:
            data = np.reshape(data, [], Nt)
        
        # Crop data to synchpts if necessary. 
        keep = np.logical_and(info_in['pairs']['r2d'] < 20, info_in['pairs']['WL'] == 2)
        foo = np.squeeze(data[keep,:])
        foo = tx4m.highpass(foo, 0.02, info_in['system']['framerate']) # bandpass filter, omega_hp = 0.02
        foo = tx4m.lowpass(foo, 1, info_in['system']['framerate']) #bandpass filter, omega_lp = 1
        foo = foo - np.roll(foo, 1, 1)
        foo[:,0] = 0
        foob = matlab.rms_py(foo) #uses new rms that only takes one input, calculates rms for every column in rms_input and outputs row vector
        NtGV = Nt - GVwin
        NtGV_mat = np.ones((1,NtGV), dtype = np.int8) # 

        if NtGV > 1: # sliding window to grab a meaningful set for 'quiet'
            GVTD_win_means = np.zeros(NtGV, order = 'F')
            i = 0
            while i <= (NtGV - 1):
                GVTD_win_means[i] =  np.mean(foob[i:((i+1)+ GVwin - 1)])
                i = i+1
            t0 = np.where(GVTD_win_means == np.min(GVTD_win_means)) # find min and set t0 --> tF
            tF = t0[0][0] + GVwin #- 1 #remove "-1" because of 0 indexing causing STD to be the wrong size #need to index t0 in order to get integer bc t0 is an array
            STD = np.std(data[:, t0[0][0]:tF], 1, ddof= 1) # Calulate STD, make sure ddof param is set = 1 so that np.STD behaves the same as matlab STD
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
            STD = np.std(data[:, t0:tF], 1, ddof=1) # Calulate STD.
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

   
    def normcND(data):
        """ 
        This function returns a column-normed matrix. It is assumed that the matrix is 2D.
        """
        vecnorm = lna.norm(data, ord = 2, axis = 0)
        data = data / vecnorm

        return data

   
    def normrND(data):
        """
        This function returns a row-normed matrix. It is assumed that the matrix is 2D. Updated for broader compatability.
        """
        dataNorm = np.sqrt(np.sum(data**2))
        data = data / dataNorm
        data[np.argwhere(np.isfinite(data) == False)] = 0

        return data
