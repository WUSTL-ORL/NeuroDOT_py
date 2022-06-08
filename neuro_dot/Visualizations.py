# General imports
import sys
import math

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import scipy.interpolate
import numpy.matlib as mlb
import numpy.matlib as nm
import functools as ft
import copy

from math import trunc
from pickle import NONE
from numpy import float64, matrix
from numpy.lib.shape_base import expand_dims
from matplotlib.pyplot import colorbar, colormaps, tight_layout
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D

# Import toolbox
from Spatial_Transforms import sx4m
from Temporal_Transforms import tx4m
from Light_Modeling import lmdl
from File_IO import io
from Analysis import anlys
from Matlab_Equivalent_Functions import matlab



class viz:
    
    
    def adjust_brain_pos(meshL, meshR, params = None):
        """
        ADJUST_BRAIN_POS Repositions mesh orientations for display.

        [Lnodes, Rnodes] = ADJUST_BRAIN_POS(meshL, meshR) takes the left and
        right hemispheric meshes "meshL" and "meshR", respectively, and
        repositions them to the proper perspective for display.

        [Lnodes, Rnodes] = ADJUST_BRAIN_POS(meshL, meshR, params) allows the
        user to specify parameters for plot creation.

        "params" fields that apply to this function (and their defaults):
            ctx         'std'       Defines inflation of mesh.
            orientation 't'         Select orientation of volume. 't' for
                                    transverse, 's' for sagittal.
            view        'lat'       Sets the view perspective.

        Dependencies: ROTATE_CAP, ROTATION_MATRIX.

        See Also: PLOTLRMESHES.
        """
        ## Parameters and Initialization
        if params == None:
            params = {}

        dy = -5

        try:
            params['ctx'] #see if params['ctx'] exists
        except KeyError:
            params['ctx'] = 'std' #if it doesn't, create and set to 'std'
        if np.logical_and('ctx' in params, not params['ctx']): #see if params['ctx'] exists and is empty
            params['orientation'] = 'std'
        try: 
            params['orientation'] #see if params['orientation'] exists
        except KeyError:
            params['orientation'] = 't' #if it doesn't, create and set to 't'
        if np.logical_and('orientation' in params, not params['orientation']): #see if params['ctx'] exists and is empty
            params['orientation'] = 't'
        try:
            params['view'] #see if params['view'] exists
        except KeyError:
            params['view'] = 'lat' #if it doesn't, create and set to 'lat'
        if np.logical_and('view' in params, not params['view']): #see if params['ctx'] exists and is empty
            params['view'] = 'lat'


        ## Choose inflation.
        ## Writing this as 3 if statements for params.ctx instead of switch/case
        if params['ctx'] == 'std':
            str = 'nodes'
        if params['ctx'] == 'inf':
            str = 'Inodes'
        if params['ctx'] == 'vinf':
            str = 'VInodes'
        Lnodes = meshL[str]
        Rnodes = meshR[str]


        ## Standardize orientation.
        ## using == in place of strcmp bc we want to see if the two strings are IDENTICAL
        ## using if statements for params.orientation instead of switch/case
        if params['ctx'] == 'std':
            if params['orientation'] == 's':
                Nln = Lnodes.shape[0]
                temp = np.concatenate((Lnodes, Rnodes), axis = 0)
                # Error_temp_in = cpm.compare2mat(temp, temp_in_matlab) #identical
                temp = RC.rotate_cap(temp, [-90, 0, 90])
                Lnodes = temp[0:Nln, :]
                fin = temp.shape[0] - 1
                Rnodes = temp[Nln:None, :] #slice from index N to index None to include the final element of the array you're trying to slice. slicing from N to -1 will exclude the final elements
            if params['orientation'] == 'c':
                None
                # Not yet supported.
                #May build this in later, but will be a pain (Ari's note from Adam)
            if params['orientation'] == 't':
                None
                # Not yet supported
                #'t' is already really std so no need to build this in (Ari's note from Adam)



        ## Normalize L and R nodes to their max and min, respectively.
        ## using if statements for params.ctx instead of switch/case
        if params['ctx'] == 'std':
            None
        if params['ctx'] == 'inf':
            Lnodes[:, 0] = Lnodes[:, 0] - max(Lnodes[:, 0])
            Rnodes[:, 0] = Rnodes[:, 0] - min(Rnodes[:, 0])
        if params['ctx'] == 'vinf':
            Lnodes[:, 0] = Lnodes[:, 0] - max(Lnodes[:, 0])
            Rnodes[:, 0] = Rnodes[:, 0] - min(Rnodes[:, 0])


        ## Rotate if necessary
        if np.logical_or(params['view'] == 'lat', params['view'] == 'med'):

            cmL = np.mean(Lnodes, axis = 0)
            cmL.shape = (1, cmL.shape[0]) #reshape to 2D row vector
            cmR = np.mean(Rnodes, axis = 0)
            cmR.shape = (1, cmR.shape[0]) #reshape to 2D row vector
            rm = RM.rotation_matrix('z', np.pi)

            if params['view'] == 'lat':
                Rnodes = (Rnodes - (mlb.repmat(cmR, Rnodes.shape[0], 1))) @ rm + (mlb.repmat(cmR, Rnodes.shape[0], 1))
            if params['view'] == 'med':
                Lnodes = (Lnodes - (mlb.repmat(cmL, Lnodes.shape[0], 1))) @ rm + (mlb.repmat(cmL, Lnodes.shape[0], 1))

            Rnodes[:, 0] = Rnodes[:, 0] + (cmL[:, 0] - cmR[:, 0])
            Rnodes[:, 1] = Rnodes[:, 1] - max(Rnodes[:, 1]) + min(Lnodes[:, 1]) + dy

        return Lnodes, Rnodes


    def applycmap(overlay, underlay, params):
        """  
        APPLYCMAP Performs color mapping and fuses images with anatomical
        models.
        
        mapped = APPLYCMAP(overlay) fuses the N-D image array "overlay" with a
        default flat gray background and applies a number of other default
        settings to create a scaled and colormapped N-D x 3 RGB image array
        "mapped".
        
        mapped = APPLYCMAP(overlay, underlay) fuses the image array "overlay"
        with the anatomical atlas volume input "underlay" as the background.
        
        mapped = APPLYCMAP(overlay, underlay, params) allows the user to
        specify parameters for plot creation.
        
        "params" fields that apply to this function (and their defaults):
            TC          0               Direct map integer data values to
                                        defined color map ("True Color").
            DR          1000            Dynamic range.
            Scale       (90% max)       Maximum value to which image is scaled.
            PD          0               Declares that input image is positive
                                        definite.
            Cmap                        Color maps.
                .P      'jet'           Colormap for positive data values.
                .N      (none)          Colormap for negative data values.
                .flipP  0               Logical, flips the positive colormap.
                .flipN  0               Logical, flips the negative colormap.
            Th                          Thresholds.
                .P      (25% max)       Value of min threshold to display
                                        positive data values.
                .N      (-Th.P)         Value of max threshold to display
                                        negative data values.
            BG          [0.5, 0.5, 0.5] Background color, as an RGB triplet.
            Saturation  (none)          Field the size of data with values to
                                        set the coloring saturation. Must be 
                                        within range [0, 1].
        
        See Also: PLOTSLICES, PLOTINTERPSURFMESH, PLOTLRMESHES, PLOTCAPMEANLL.
        """
    #  Parameters and initialization.
        img_size = overlay.shape
        overlay = np.transpose(overlay.flatten()) #+ np.transpose(overlay.flatten()) + notation in matlab does not indicate addition   #% Make sure not logical 
        mapped = np.zeros((len(overlay), 3))
        overlay[np.argwhere(np.isfinite(overlay)==False)] = 0
        if underlay != []:
            underlay[np.argwhere(np.isfinite(underlay)==False)]=0

        if not sum(abs(overlay[overlay!=0])):
            print(['The Overlay has only elements equal to zero'])
            mapped = mapped + 0.5
            map_out=[0.5,0.5,0.5]

            return
        
        if underlay is None  or underlay == []:
            params['underlay'] = 0
        else:
            params['underlay'] = 1
            underlay = np.transpose(underlay.flatten())

            max_underlay = max(underlay)
            underlay = [number / max_underlay for number in underlay]  # come back to this later when we have an input where underlay is not empty - everything for the first pipeline uses an empty underlay

        if params is None:
            params = {}


        if 'PD' not in params or params['PD'] == []:
            params['PD'] = 0

        if 'TC' not in params or params['TC'] == []:
            params['TC'] = 0

        if 'DR' not in params or params['DR'] == []:
            params['DR'] = 1000

        if 'Scale' not in params or params['Scale'] == []:
            params['Scale'] = 0.9 * max(overlay)

        if 'Th' not in params or params['Th'] == []:
            params['Th']['P'] = 0.25 * params['Scale']
            params['Th']['N'] = -params['Th']['P']

        if 'N' not in params['Th'] or params['Th']['N'] == []:
            params['Th']['N'] = -params['Th']['P']

        if ('Cmap' not in params or params['Cmap'] == []) or ('P' in params['Cmap'] and params['Cmap']['P'] == []):
            params['Cmap']['P'] = 'jet'
        else:
            if not isinstance(params['Cmap'], dict):
                temp = params['Cmap']
                params['Cmap'] = {}
                params['Cmap']['P'] = temp
        
        Cmap = {}                                # Initialize colormap as an empty python dictionary
        if isinstance(params['Cmap']['P'], str): # Generate positive and negative Cmaps, if present (chars are represented in Python as length 1 strings)
            Cmap['P'] = plt.get_cmap(str(params['Cmap']['P']),params['DR'])
            Cmap['P'] = Cmap['P'](range(params['DR']))
        elif params['Cmap']['P'].isnumeric():
            Cmap['P'] = params['Cmap']['P']
        if 'flipP' in params['Cmap']  and params['Cmap']['flipP'] != []  and params['Cmap']['flipP']: # Optional colormap flip.
            Cmap['P'] = np.flip(Cmap['P'], 0)

        if 'N' in params['Cmap'] and params['Cmap']['N'] != []:
            if isinstance(params['Cmap']['N'], str):
                Cmap['N'] = eval(params['Cmap']['N'], '[', str(params['DR']), ']')
            elif params['Cmap']['N'].isnumeric():
                Cmap['N'] = params['Cmap']['N']
            
            if 'flipN' in params['Cmap'] and params['Cmap']['flipN'] != [] and params['Cmap']['flipN']:
                Cmap['N'] = np.flip(Cmap['N'], 0)
            
            params['PD'] = 1

        if 'BG' not in params or params['BG'] == []:
            params['BG'] = [0.5, 0.5, 0.5]

        ## Get Background.
        bg = np.argwhere(overlay == 0)
        ## Truecolor - this wasn't properly tested because the sample data didn't require entering this loop; we will be able to test the else! (hopefully)
        if params['TC']:
            # Color in RGB channels.
            Nc = np.shape(Cmap['P'], 0)
            for i in range(0,Nc):
                mapped[np.where(overlay == i)[0], 0] = Cmap['P'][i, 0]
                mapped[np.where(overlay == i)[0], 1] = Cmap['P'][i, 1]
                mapped[np.where(overlay == i)[0], 2] = Cmap['P'][i, 2]

            #Color in background.
            if not params['underlay']:
                mapped[bg, :] = nm.repmat(params['BG'], len(bg), 1)

            else:
                mapped[bg, :] = [underlay[bg] * params.BG[1], underlay[bg] * params['BG'][2], underlay[bg] * params['BG'][3]]  ## possible cause of error

        else:
            ## Label Data Outside Thresholds to Background.
            bg = ft.reduce(np.union1d,(bg, np.intersect1d(np.argwhere(overlay <= params['Th']['P']), np.argwhere(overlay >= params['Th']['N']))))
            ##  Scaling and Dynamic Range.
            # All below threshold set to gray, all above set to colormap.
            if params['PD']: #% Treat as pos def data?:
                overlay = (np.multiply(np.divide(overlay, params['Scale']), (params['DR'])))# % Normalize and scale to colormap
                fgP = np.argwhere(overlay > 0) # % Populate Pos to color; returns array of indices - potential cause of off by one errors later on
                fgN = np.argwhere(overlay < 0) #% Populate Neg to color
            else:
                overlay = (np.multiply(np.divide(overlay, params['Scale']) , params['DR'] / 2))# % Normalize and scale to colormap
                overlay = overlay + (params['DR'] / 2) #% Shift data
                fgP = np.argwhere(overlay != (params['DR'] / 2))   # % Populate Pos to color
                overlay[np.argwhere(overlay <= 0)] = 1 #% Correct for neg clip
            
            overlay[np.argwhere(overlay >= params['DR'])] = params['DR']# % Correct for pos clip
            
            
            
            #%% Color in RGB channels.
            mapped[fgP, 0] = Cmap['P'][np.ceil(overlay[fgP]).astype(int)-1, 0]
            mapped[fgP, 1] = Cmap['P'][np.ceil(overlay[fgP]).astype(int)-1, 1]
            mapped[fgP, 2] = Cmap['P'][np.ceil(overlay[fgP]).astype(int)-1, 2]

            #% Pos def treatment.
            if params['PD']:
                if 'N' in params['Cmap']:
                    overlay[-overlay >= params['DR']] = -params['DR']# % Correct for clip.
                    mapped[fgN, 0] = Cmap['N'][np.ceil(-overlay[fgN]), 0]
                    mapped[fgN, 1] = Cmap['N'][np.ceil(-overlay[fgN]), 1]
                    mapped[fgN, 2] = Cmap['N'][np.ceil(-overlay[fgN]), 2]

                else:
                    bg = np.union1d(bg, fgN) #% If only pos, neg values to background.

            ## %% Apply background coloring.
        if not params['underlay']:# not in params:
            mapped[bg, :] = nm.repmat(params['BG'], len(bg), 1)
        else:
            mapped[bg, :] = [np.multiply(underlay[bg], params['BG'][0]), np.multiply(underlay[bg], params['BG'][1]), np.multiply(underlay[bg], params['BG'][2])] ## possible cause of error

        ##%% Apply Saturation if available
        # Saturation is not available in any of the uses in NeuroDOT so far that we can see
        if 'Saturation' in params:#isfield(params,'Saturation')
            nbg = np.setdiff1d(np.arange(1, np.shape(mapped)[0]), bg)
            if not params['underlay']: #not in params:# ~params.underlay
                mapped[nbg,:] = (np.multiply(params['Saturation'][nbg], mapped[nbg,:])) + (np.multiply(params['BG']* (1- params['Saturation'][nbg])))

            else:
                mapped[nbg,:] = (np.multiply(params['Saturation'][nbg], mapped[nbg,:])) + (np.multiply(params['BG'], (np.multiply(1- params['Saturation'][nbg],underlay[nbg, :]))))
            
            # Reshape to original size + RGB.
        if (len(img_size) == 1):
            mapped = np.reshape(mapped, [img_size[0], 3])
        else:
            mapped = np.reshape(mapped, list(img_size) + [3])

        ## Create a final color map.
        if not params['underlay']:
            thresh_zone_color = params['BG']#  Same color as background if no underlay.
        else:
            thresh_zone_color = [0, 0, 0]# Black if there is an underlay.

        map_out = Cmap['P']
        if params['TC']:
            map_out = params['Cmap']['P']
        else:
            if params['PD']:
                thP = np.floor(params['Th']['P'] / params['Scale'] * params['DR']).astype(int) 
    
                map_out[1:thP, 0:3] = nm.repmat(thresh_zone_color, thP, 1)
                if 'N' in Cmap:
                    thN = np.floor(params['Th']['N'] / params['Scale'] * params['DR'])
                    tempN = Cmap['N']
                    tempN[1:thN, :] = nm.repmat(thresh_zone_color, thN, 1)
                    map_out = np.hstack(np.flip(tempN, 0), map_out)
                
            else:
                thP = np.floor(params['Th']['P'] / params['Scale'] * params['DR'] / 2 + (params['DR'] / 2)).astype(np.int64)
                thN = np.ceil(params['Th']['N'] / params['Scale'] * params['DR'] / 2 + (params['DR'] / 2)).astype(np.int64)
                if thN<1:
                    thN=1
                thresh_A = nm.repmat(thresh_zone_color, thP-thN+1, 1)
                thresh_A = np.append(thresh_A, 1)    
                map_out[thN:thP, :] = thresh_A

        
        newcmp = colors.ListedColormap(map_out)
        map_out = newcmp

        return mapped,map_out,params

    
    def DrawColoredSynchPoints(info_in, subplt, SPfr = 0):
        """
        This function draws vertical lines over a plot/imagesc type axis to
        delineate synchronization points on time traces.
        This function assumes the standard NeuroDOT info structure and paradigm
        format: 
        info.paradigm       Contains all paradigm timing information
        *.synchpts          Contains sample points of interest
        *.synchtype         Optional field that contains marker that can
                                distinguish between synchronization types,
                                traditionally as sound boops of differing
                                frequencies, i.e., 25, 30, 35 (Hz).
        *.Pulse_1           Conventionally used to index synchpts that
                                denote 'bookends' of data collection as
                                well as resting (or OFF) periods of a given
                                stimulus.
        *.Pulse_2           Conventionally used to index synchpts that
                                correspond to the start of a given stimulus
                                epoch (e.g., the start of a flickering
                                checkerboard)
        *.Pulse_3           Index of synchpts of a 2nd stimulus type
        *.Pulse_4           Index of synchpts of a 3rd stimulus type
        The 2nd input, SPfr, selects whether or not to adjust synchpoint timing
        based on framerate of data. (Default=0).
        """
        ## Parameters and Initialization.
        if 'paradigm' not in info_in:
            return 
        if 'synchpts' not in info_in["paradigm"]:
            return 
        h = subplt.axes

        xLim=h.get_xlim()
        yLim=h.get_ylim()

        # if SPfr is None: #unused bc SPfr is set optionally in function definition
        #     SPfr = 0

        if SPfr: 
            fr=info_in["system"]["framerate"]
            synchs=info_in["paradigm"]["synchpts"]/fr
        else:
            synchs=info_in["paradigm"]["synchpts"]


        for i in range(1, 5):
            key = "Pulse_" + str(i)
            if key not in info_in["paradigm"]:
                info_in["paradigm"][key]=[]

        #  Draw lines
        for i in range(1,len(synchs)+1): #% Draw synch pt bars
            if i in info_in["paradigm"]["Pulse_1"]:
                subplt.plot(np.array([1,1]) * np.array(synchs[i-1]),yLim, color="red",linewidth = 1.5, alpha=1)
            elif i in info_in["paradigm"]["Pulse_2"]:
                subplt.plot(np.array([1,1]) * np.array(synchs[i-1]),yLim, color="green",linewidth = 1.5, alpha=1)
            elif i in info_in["paradigm"]["Pulse_3"]:
                subplt.plot(np.array([1,1]) * np.array(synchs[i-1]),yLim, color="b", alpha=1)
            elif i in info_in["paradigm"]["Pulse_4"]:
                subplt.plot(np.array([1,1]) * np.array(synchs[i-1]),yLim, color="m", alpha=1)
            else: 
                if h.get_color() is not None:
                    subplt.plot(np.array([1,1]) * np.array(synchs(i)),yLim, color="k",alpha=1)
                else:
                    subplt.plot(np.array([1,1]) * np.array(synchs(i)),yLim, color="w", alpha=1)


    def nlrGrayPlots_220324(nlrdata, info, mode = 'auto'):
        """
        This function generates a gray plot figure for measurement pairs
        for just clean WL==2 data. It is assumed that the input data is
        nlrdata that has already been filtered and resampled. The data is grouped
        into info.pairs.r2d<20, 20<=info.pairs.r2d<30, and 30<=info.pairs.r2d<40.
        """
    ## Parameters and Initialization
        Nm = np.shape(nlrdata)[0]
        Nt = np.shape(nlrdata)[1]

        LineColor = 'w'
        BkndColor = 'k'
        if 'GVTD' in info:
            Nrt = np.shape(info['GVTD'])[0]
        
        fig = plt.figure(figsize = [9,5], facecolor = BkndColor)
        wl = np.unique(info['pairs']['lambda'][info['pairs']['WL'] == 2])[0]
        

        try: 
            info['MEAS']
        except KeyError:
            info['MEAS']['GI'] = np.ones([np.shape(info['pairs']['src'][0]), 1])
        
        ## Prepare data and imagesc together
        #keep.d1, keep.d2, keep.d3 are logical arrays of ones and zeros, not T/F 
        # ex: for keep.d1, (info.MEAS.GI and (info.pairs.r2d <20)) and (info.pairs.WL == 2)
        # in python this is accomplished with a nested logical_and function call which outputs a boolean array
        # the boolean array is changed to a ones and zeros array using .astype()
        keep = {}
        keep['d1'] = np.logical_and(np.logical_and(info['MEAS']['GI'],(info['pairs']['r2d']<20)),(info['pairs']['WL'] == 2)).astype(np.uint8)
        keep['d2'] = np.logical_and(np.logical_and(np.logical_and(info['MEAS']['GI'],(info['pairs']['r2d']>=20)), #and
        (info['pairs']['r2d']<30)), #and
        (info['pairs']['WL'] == 2)).astype(np.uint8)
        keep['d3'] = np.logical_and(np.logical_and(np.logical_and(info['MEAS']['GI'],(info['pairs']['r2d']>=30)), #and
        (info['pairs']['r2d']<40)), #and
        (info['pairs']['WL'] == 2)).astype(np.uint8)
        
        SepSize = np.round((np.sum(keep['d1']) + np.sum(keep['d2']) + np.sum(keep['d3']))/50)
        nans = np.empty((SepSize.astype(np.int64), Nt))
        nans[:] = np.nan
        data1 = np.concatenate((np.squeeze(nlrdata[np.argwhere(keep['d1'])]), nans, 
        np.squeeze(nlrdata[np.argwhere(keep['d2'])]), nans, 
        np.squeeze(nlrdata[np.argwhere(keep['d3'])])), axis = 0)
        M = np.nanstd((data1[:]),ddof=1)*3


        # fig = plt.subplots(dpi = 100, facecolor = 'black')
        if 'GVTD' in info:
            gs = gridspec.GridSpec(3,3)
            ax1 = plt.subplot(gs[0,:])
            ax2 = plt.subplot(gs[1:,:])
        else:
            gs = gridspec.GridSpec(1,3)
            ax2 = plt.subplot(gs[0,:])

        ## Line Plot of  #will cause issues, but currently isn't used
        if 'GVTD' in info:
            ax1.plot(np.array(range(0, Nrt)), info['GVTD'], linewidth = 2, color = 'r')
            ax1.set_xlim(0, Nrt)
            ax1.set_facecolor([0,0,0,0])
            ax1.spines[:].set_color(LineColor)
            ax1.tick_params(axis = 'x', colors = LineColor)
            ax1.tick_params(axis = 'y', colors = LineColor)
            ax1.locator_params(axis = 'x', nbins = 10)
            ax1.locator_params(axis = 'y', nbins = 5) #does not show top tick of 0.02
            ax1.set_title('GVTD', color = 'w')
            ax1.set_ylabel('a.u', color = 'w')
            plt.tight_layout()
        
        ## Gray Plot data
        ax2.set_facecolor([0,0,0,0])
        ax2.set_autoscale_on(True)
        ax2.imshow(data1,vmin = -1*M, vmax = 1*M, cmap = 'gray', aspect = 'auto') #, extent = [-1,5.5,-0.5,3.1])

        # Plot Synchs
        if mode == 'auto':
            viz.DrawColoredSynchPoints(info, ax2) #if using an image with DCSP, set SPfr to 0 or don't set SPfr to anything bc it's auto set in DCSP
        Npix = np.size(data1,0)

        # Plot separators #not sure if this is necessary as values were set manually in lines 103-105
        dz1 = len(keep['d1'])
        dz2 = len(keep['d2'])
        dz3 = len(keep['d3'])
        dzT = dz1+dz2+dz3+2*SepSize

        # Add labels
        ax2.set_title('\u0394 ' + str(wl) + ' nm', color = 'white') #use unicode and \u to get symbols
        if 'GVTD' in info:
            ax2.set_ylabel('Rsd: [1,20) mm     Rsd: [20,30) mm     Rsd: [30,40) mm', rotation = 90.0, color = 'w', fontsize = 7, va = 'center', labelpad= -10)
        else:
            ax2.set_ylabel('Rsd: [1,20) mm             Rsd: [20,30) mm             Rsd: [30,40) mm', rotation = 90.0, color = 'w', fontsize = 7, va = 'center', labelpad= -10)
      
    def Plot_RawData_Cap_DQC(data,info_in,params = None):
        """ 
        This function generates plots of data quality as related to the cap
        layout including: relative average light levels for 2 sets of distances
        of source-detector measurements, the cap good measurements plot, and 
        a measure of the pulse power at each optode location.
        """   
        if params == None:
            params = {} # if we define params as {} it is mutable, none cannot be modified, not global, scoped to function def
        plt.rcParams["font.size"] = 8
        # %% Parameters and Initialization
        if 'bthresh' not in params:
            params['bthresh'] = 0.075
        if 'rlimits' not in params:
            params['rlimits'] = np.stack(([1,20],[21,35],[36,43]))
        elif len(params['rlimits']) == 1:
            params['rlimits'] =  np.stack((params["rlimits"], [21,35],[36,43]))
        elif len(params['rlimits']) == 2:
            params['rlimits'] =  np.stack((params["rlimits"], [36,43]))

        Rlimits = params['rlimits']
        if 'mode' not in params:
            params['mode'] = 'good' # changed from 'good' since 'good' doesn't work with PlotCapData
        if 'useGM' not in params:
            params['useGM'] = 1
        if not np.all(np.isreal(data)):
            data = abs(data)

        # %% Check for good measurements
        if 'MEAS' not in info_in or 'GI' not in info_in['MEAS']:
            data_out, Phi_0 = tx4m.logmean(data)
            info_out = anlys.FindGoodMeas(data_out,info_in, params['bthresh'])
        else:
            info_out = info_in.copy()

        fig = plt.subplots(dpi = 30, facecolor = 'black', figsize = (40,30))
        gs = gridspec.GridSpec(4,2, height_ratios=[5,5,2.5,2.5], width_ratios=[1,1])
        ax1 = plt.subplot(gs[0,0], aspect = 'equal')
        ax2 = plt.subplot(gs[0,1], aspect = 'equal')
        ax3 = plt.subplot(gs[1,0], aspect = 'equal')
        ax4 = plt.subplot(gs[1,1], aspect = 'equal')
        ax5 = plt.subplot(gs[2:,:], aspect = 'equal')

        # %% Mean signal level at each optode
        params2 = params.copy()
        if len(params2['rlimits']) > 1:
            params2['rlimits'] = Rlimits[0,:]
        else:
            params2['rlimits'] = Rlimits[0]

        
        info2 = info_out.copy()
        
        # I think that params['mode'] is getting changed to 'good' but it needs to be 'patch for plotcapdata to return x an dy to plotcapmeanll

        plt.rcParams['lines.linewidth'] = 0.1
        params6 = params2.copy()

        dataout = viz.PlotCapMeanLL(data,info2,ax1,params6)

        if len(params2['rlimits']) > 1:
            params2['rlimits'] = Rlimits[1,:]
        else:
            params2['rlimits'] = Rlimits[1]
        params2['rlimits'] = Rlimits[1,:]
    
        params5 = params2.copy()
        info5 = info_out.copy()
        dataout2 = viz.PlotCapMeanLL(data, info5, ax2,params5)

        info_out['MEAS']['Phi_o'] = dataout2
        info6 = info_out.copy()
        
    ##  Good (and maybe bad) measurements
        params7 = copy.deepcopy(params2)
        params7['rlimits']=[np.amin(Rlimits),np.amax(Rlimits)]
        viz.PlotCapGoodMeas(info6, ax5, params7)

        ## Cap Physiology Plot
        del params5['mode']
        info3 = info_out.copy()
        params5['rlimits']=Rlimits[0,:]
        Plevels1 = viz.PlotCapPhysiologyPower(data, info3, ax3, params5)
        info_out['MEAS']['Pulse_SNR_R1'] = Plevels1

        info4 = info_out.copy()
        params4 = params2.copy()
        params4['rlimits']=Rlimits[1,:]
        Plevels2 = viz.PlotCapPhysiologyPower(data, info4, ax4, params4)
        info_out['MEAS']['Pulse_SNR_R2'] = Plevels2

        return info_out

    
    def defineticks(start, end, steps, blankLabels=False): 
        """ 
        This function generates a single-page report that includes:
        light fall off as a function of Rsd
        SNR plots vs. mean light level
        source-detector mean light-level plots
        Power spectra for 830nm at 2 Rsd
        Histogram for measurement noise

        See: Plot_RawData_Metrics_I_DQC
        """
        ticks = []
        numsteps = (end - start)//steps
        for i in range(int(numsteps)+1):
            x = start + steps*i
            if blankLabels:
                ticks.append(' ')
            else:
                ticks.append(x)

        return ticks


    def Plot_RawData_Metrics_I_DQC(data,info,params = None):
        """
        This function generates a single-page report that includes:
            light fall off as a function of Rsd
            SNR plots vs. mean light level
            source-detector mean light-level plots
            Power spectra for 830nm at 2 Rsd
            Histogram for measurement noise
        """
        #%% Parameters and Initialization
        if params == None:
            params = {}

        paramskeys = params.keys() 
        if "bthresh" not in paramskeys: 
            params["bthresh"]=0.075
            
        paramsrlimts = [[1,20],[21,35],[36,43]]
        if "rlimits" not in paramskeys: 
            params["rlimits"]=paramsrlimts
        elif len(params["rlimits"]) == 1: 
            concat = params["rlimits"] + paramsrlimts[1:]
            params["rlimits"] = concat
        elif len(params["rlimits"][1]) == 2: 
            params["rlimits"] = params["rlimits"] + paramsrlimts[2:]
            
        if "logfft" not in paramskeys: 
            params["logfft"]=0

        if "LFO_GI" not in paramskeys:
            params["LFO_GI"] = 0
            
            
        
        infosystemkeys = info['system'].keys() 
        if "init_framerate" in infosystemkeys: 
            fr=info['system']['init_framerate']
        else:
            fr=info['system']['framerate']
    
        wls=np.unique(info['pairs']['lambda'])
        Nwls=len(wls);
        wavelengths=[0]*Nwls
        for i in range(Nwls):
            wavelengths[i]=str(wls[i]) + ' nm'
        Ns=np.amax(info['pairs']['Src'])
        Nd=np.amax(info['pairs']['Det'])
        [Nm,Nt]=np.shape(data)
        
        if not np.isreal(data).all():
            data=abs(data)
        #casting to an int but may not be correct for all cases 
        Nm=int(Nm/Nwls)

        if Nt<(60/fr):
            ti=1
            tf=Nt
        elif Nt:
            ti=round(Nt/2)-round(5*fr)
            if ti<1:
                ti=1
            tf=round(Nt/2)+round(5*fr)
            if tf>=Nt:
                tf=Nt

        # % Add in more exact correction for APD Sensitivity (~6e6 V/W) to convert Mags
        # % to Voltage to OptPower(rms). Also include estimate for theoretical noise
        # % floor via NEP*(sqrt(bandwidth))
        if "Input_Refer_Power" not in paramskeys:
            params["Input_Refer_Power"] = 1

        if params["Input_Refer_Power"]:
            if "Sens" in paramskeys:
                APDsens=params["Sens"]
            else:
                APDsens=6e6# % ~6 MV/W is sensitivity of our APDs
            data=(data/APDsens)*1e6# % Input refer data to optical power in micro Watts

        if "NEPth" not in paramskeys:
            params["NEPth"] = 0
        if params["NEPth"]:
            NEPth = params["NEPth"]
            # Theoretical noise floor should look like NEP*sqrt(bw)
        #     %% Check for good measurements
        infokeys = info.keys()
        if 'MEAS' in infokeys: 
            infoMeaskeys = info['MEAS'].keys()
            if 'GI' not in infoMeaskeys: 
                ##NOTE FOR LOGMEAN THERE ARE 2 ouputs so you have to choose which ones!!! 
                info = anlys.FindGoodMeas(tx4m.logmean(data)[0], info, params['bthresh'])
        else: 
            ##NOTE FOR LOGMEAN THERE ARE 2 ouputs so you have to choose which ones!!! 
            info = anlys.FindGoodMeas(tx4m.logmean(data)[0], info, params['bthresh'])

        # Set Up Figures 
        Phi_0=np.ndarray.mean(data,1)
        M=np.ceil(np.amax(np.log10(data)))
        yM=10**M
        where = np.where(info['pairs']['lambda'] == wls[0])[0]
        r=info['pairs']['r3d'][where]
        col = len(Phi_0)//Nm
        plt.rcParams.update({'font.size': 4})

        fig = plt.figure(dpi = 400)#, figsize = (3,2))#, tight_layout = True)
        gs = gridspec.GridSpec(3,6, width_ratios = [1,1,1,1,1,1], height_ratios = [0.5,0.5,0.75], hspace = 0.75, wspace = 1.2)#, dpi = 75)
        ax1 = plt.subplot(gs[0:2,0:2])
        ax2 = plt.subplot(gs[2,0:2])
        ax3 = plt.subplot(gs[0,2])
        ax4 = plt.subplot(gs[0,3])
        ax5 = plt.subplot(gs[1,2])
        ax6 = plt.subplot(gs[1,3])
        ax7 = plt.subplot(gs[2,2])
        ax8 = plt.subplot(gs[2,3])
        ax9 = plt.subplot(gs[0:2,4:6])
        ax10 = plt.subplot(gs[2,4:6])

        plt.gcf().set_facecolor('black')

        # Plot 1: Light Level FallOff
        array = np.reshape(Phi_0,(Nm, len(Phi_0)//Nm), order='F')
        ax1.semilogy(r, array, '.', markersize=0.5)
        ax1.semilogy(r, array, '.', markersize=0.5)
        
        # Appearance
        ax1.set_facecolor('black')

        # Axis limits 
        ax1.set_xlim([0,100])
        ax1.set_ylim([1e-7,yM])
        
        # Ticks and Gridlines 
        ax1.grid(axis="x", linestyle='-', linewidth=0.15)
        ax1.grid(axis="y", linestyle='dotted', linewidth=0.4, which='both')
        ax1.spines['bottom'].set_color('white')
        ax1.spines['top'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['right'].set_color('white')
        ax1.tick_params(axis='x', colors='white', length = 1, width = 0.5, direction = 'in')
        ax1.tick_params(axis='y', colors='white', length = 1, width = 0.5, direction = 'in')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.set_xticks(np.arange(0, 101, 10))
        
        # Axis Labels
        ax1.set_xlabel('Source-Detector Separation ( mm )', color='white')
        ax1.set_ylabel('$\u03A6_{0}$ ( \u03bcW )', color='white')
        
        # Legend 
        ax1.legend(wavelengths)

        # Clipping
        if "Clipped" in info["MEAS"]:
            keep = np.where(info["MEAS"]["Clipped"]  == 1)[0]
            plt.semilogy(info["pairs"]["r3d"][keep],Phi_0[keep],'x')
            ax1.legend(wavelengths, "Clipped")

    ## Plot 2: <fft> for max wavelength at 2 distances
        lmdata = tx4m.logmean(data)[0]
        keep = np.logical_and(np.logical_and(np.logical_and((info["pairs"]["lambda"]==np.max(wls)) , info["MEAS"]["GI"]) , info["pairs"]["r3d"] >= params["rlimits"][0][0]) , info["pairs"]["r3d"]<=params["rlimits"][0][1])
        r1=np.mean(info["pairs"]["r3d"][keep])
        #% WL2reg_a=mean(squeeze(lmdata(keep,:)),1);
        WL2reg_a=lmdata[keep]
        ftdomain, ftmag0,_,_ = tx4m.fft_tts(WL2reg_a,fr)
        ftmag=matlab.rms_py(ftmag0)
        ax2.semilogx(ftdomain,ftmag,'--', color = 'red', linewidth = 1)
        
        keep=np.logical_and(np.logical_and(np.logical_and((info["pairs"]["lambda"] == np.max(wls)) , info["MEAS"]["GI"]) , info["pairs"]["r3d"] >= params["rlimits"][1][0]) , info["pairs"]["r3d"] <= params["rlimits"][1][1])
        r2=np.mean(info["pairs"]["r3d"][keep])
        #% WL2reg_b=mean(squeeze(lmdata(keep,:)),1)
        WL2reg_b=lmdata[keep]
        ftdomain, ftmag0,_,_ = tx4m.fft_tts(WL2reg_b,fr)
        ftmag = matlab.rms_py(ftmag0)
        
        ax2.loglog(ftdomain,ftmag, '-', color = 'magenta', lineWidth = 0.5)   
        ax2.set_xlim([1e-3,fr/2])
        ax2.set_ylim([10e-7,10e-2])
        ax2.set_xlabel('Frequency [Hz]')
        ax2.xaxis.label.set_color('white')
        ax2.grid(axis="x", linestyle='--', linewidth=0.25, which='both')
        ax2.grid(axis="y", linestyle='-', linewidth=0.25)    
        ax2.set_facecolor('black')
        ax2.set_ylabel('|P1 [au]|')
        ax2.yaxis.label.set_color('white')
        ax2.tick_params(axis='x', colors='white', length = 1, width = 0.5, direction = 'in')
        ax2.tick_params(axis='y', colors='white', length = 1, width = 0.5, direction = 'in')
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.margins(y = 0)
        ax2.legend(['\u03BC('+ str(np.max(wls))+' nm, ~'+ format(r1,'.4f')+' mm)', '\u03BC('+str(np.max(wls))+' nm, ~'+format(r2,'.4f')+' mm)'], prop={'size': 3})
        
        # Plot 9: Time traces
        where1 = np.where(info['pairs']['r3d'] >= np.min(params['rlimits']), 1,0)
        where2 = np.where(info['pairs']['r3d'] <= np.max(params['rlimits']), 1, 0)
        subwhere_3 = np.where(info['pairs']['lambda'] == np.max(wls), 1, 0)
        where3 = np.logical_and(info['MEAS']['GI'],  subwhere_3)    
        
        keep = np.logical_and(where1, where2)
        keep = np.logical_and(keep, where3)    
        
        if sum(keep):
            x = np.arange(ti+1,tf+2)/fr      
            y = np.squeeze(data[keep,ti:tf+1])    

            # Plot data
            ax9.semilogy(x,y.transpose(), '-', linewidth=0.25)

            # Appearance
            ax9.set_facecolor('black')

            # Axis Limits 
            ax9.set_xlim([220,230])
            ax9.set_ylim([1e-2,1e-1])

            #Ticks and Gridlines 
            ax9.spines['bottom'].set_color('white')
            ax9.spines['top'].set_color('white')
            ax9.spines['left'].set_color('white')
            ax9.spines['right'].set_color('white')
            ax9.tick_params(axis='x', colors='white',length = 1, width = 0.5, direction = 'in')
            ax9.tick_params(axis='y', colors='white', pad = 1, length = 1, width = 0.5, direction = 'in')
            ax9.xaxis.label.set_color('white')
            ax9.yaxis.label.set_color('white')
            ax9.set_xticks(np.arange(220, 231, 1))

            # Axis Labels and Title
            ax9.set_xlabel('Time [sec]', color='white')
            ax9.set_title('\u03A6(t) ' + wavelengths[1] + 'GI: Rsd from ' + str(np.min(params['rlimits'])) + '-'+ str(np.max(params['rlimits'])) + ' mm', color='white' )
        

        
        # Plot 3: Phi_0 WL 1
        keep=np.where(info["pairs"]["lambda"] == np.min(wls),1,0)
        dList=nm.repmat(range(1,Nd+1),Ns,1)
        measFull=np.c_[nm.repmat(range(1,Ns+1),Nd,1).flatten(),dList.flatten('F')]
    

        # we only want whether the rows are included, so if both elements in each row are True, we want that index to be 1, but in a column
        # if there is a false anywhere, don't include the index of that row
        subIa = np.where(np.isin(measFull, [np.c_[info["pairs"]["Src"][keep==1],info["pairs"]["Det"][keep==1]]]),1,0)
        Ia = np.all(subIa, axis = 1) 
        Ib = np.argwhere(Ia)
        Ia = np.reshape(Ia,(Ns,Nd))
        Ib = np.reshape(Ib,(Ns,Nd))
        # Ib0 = np.argwhere(Ib == 0)  #this check for zeros wasn't necessary since there weren't any; may cause issues with later functions 
        # Ib[Ib == 0] = [] # Was unable to test if this works or not due to no zeros in data
        sdFull=np.zeros((Ns,Nd)).flatten() # Arrays are flattened here for the sake of indexing arrays in Python - this may be done more efficiently in another way 
        sdFull[Ia.flatten()]=Phi_0[Ib.flatten()]
        sdFull = np.reshape(sdFull,(Ns,Nd), order = 'F') 
        
        # Plot Data
        p0 = plt.gca()
        jet = plt.get_cmap('jet', 1000)
        ax3 = fig.add_axes([0.315, 0.675, 0.22, 0.22]) 
        ax3.xaxis.set_label_coords(.5, -0.075)
        ax3.imshow(np.log10(sdFull), cmap = jet, extent = [-0.5, 28, -0.50, 24], aspect = 28/24, vmin = -5, vmax = -1)#,[-5,-1]) # the colors aren't 100% identical to matlab, the reds are less intense but the blues seem ok. there is not much of a difference
        
        # Appearance
        jet.set_under(color='black')    

        # Axis Labels and Title
        ax3.set_title('$\u03A6_{0}$ '+ str(wls[0])+' nm', color = 'w', y = 0.95)
        ax3.set_xlabel('Detector')
        ax3.xaxis.label.set_color('white')
        ax3.set_ylabel('Source')
        ax3.yaxis.label.set_color('white')
        ax3.yaxis.set_label_coords(-0.1, 0.5)
        plt.subplots_adjust(left=0.01, right = 1)
        ax3.set_yticklabels(defineticks(20,5,-5),fontsize = 3.5)
        ax3.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)
        ax3.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax3.tick_params(axis='y', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax3.set_yticks(defineticks(4.5,19.5,5)) 
        ax3.set_xticks(defineticks(4.5,24.5,5))

        ax3.spines['bottom'].set_color('white')
        ax3.spines['top'].set_color('white')
        ax3.spines['left'].set_color('white')
        ax3.spines['right'].set_color('white')
        cMap = np.vstack(([0,0,0,0], jet(range(1000))))

        # Plot 4:  Phi_0 WL 2
        sdFull=np.zeros((Ns,Nd)).flatten() #flattening is causing issues somewhere
        sdFull[Ia.flatten()]=Phi_0[Ib.flatten()+Nm]
        sdFull = np.reshape(sdFull,(Ns,Nd), order = 'F')
        ax4 = fig.add_axes([0.475, 0.675, 0.22, 0.22]) # creates an axis of given size, but cannot pass it to the subplot axis you want
        ax4.xaxis.set_label_coords(.5, -0.075)      
        im4 = ax4.imshow(np.log10(sdFull), cmap = jet, extent = [-0.5, 28, -0.50, 24], aspect = 28/24, vmin = -5, vmax = -1)
        p0=plt.gca()
        ax4.set_title('$\u03A6_{0}$ '+ str(wls[1]) +' nm',color = 'w', y = 0.95)
        ax4.set_xlabel('Detector')
        ax4.xaxis.label.set_color('white')
        ax4.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax4.set_yticklabels(defineticks(5,25,5,blankLabels = True))
        ax4.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)

        ax4.set_xticks(defineticks(4.5,24.5,5))


        ax4.spines['bottom'].set_color('white')
        ax4.spines['top'].set_color('white')
        ax4.spines['left'].set_color('white')
        ax4.spines['right'].set_color('white')

        # this works for now but might not work for different screen sizes
        axins1 = inset_axes(ax4,
                        width="5%",  # width = 50% of parent_bbox width
                        height="80%",  # height : 5%
                        loc='lower right',
                        bbox_to_anchor=(0.12, 0.05, 1, 1),
                        bbox_transform=ax4.transAxes)
        
        cb = fig.colorbar(im4, cax=axins1, orientation="vertical", drawedges=False)
        cb.ax.yaxis.set_tick_params(color="white", pad = 0.8, length = 0)
        cb.set_ticks([-5, -3, -1])
        cb.set_ticklabels(['10e-5','\u03BCW','10e-1'])
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")
        cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0.5)

        # Plot 5:  std(Y) WL 1
        if 'synchpts' in info["paradigm"]:
            NsynchPts = len(info["paradigm"]["synchpts"])# % set timing of data
            if NsynchPts > 2:
                tF = info["paradigm"]["synchpts"][-1]
                t0 = info["paradigm"]["synchpts"][1]
            elif NsynchPts == 2:
                tF = info["paradigm"]["synchpts"][1]
                t0 = info["paradigm"]["synchpts"][0]
            else:
                tF = np.shape(data)[1]
                t0 = 0
            stdY=np.std(lmdata[:, t0:tF],1,ddof = 1)
        else:
            stdY=np.std(lmdata,1,ddof = 1)
        
        sdFull=np.zeros((Ns,Nd)).flatten() 
        sdFull[Ia.flatten()]=stdY[Ib.flatten()]
        sdFull = np.reshape(sdFull,(Ns,Nd), order = 'F')

        ax5 = fig.add_axes([0.315, 0.4, 0.22, 0.22]) 
        ax5.imshow(sdFull, cmap = jet, extent = [-0.5, 28, -0.50, 24], aspect = 28/24, vmin = 0, vmax = 0.2)
        ax5.set_title('\u03C3 (Y) '+ str(wls[0]) + ' nm', color = 'w', y = 0.95)
        ax5.set_xlabel('Detector', fontsize = 4)
        ax5.xaxis.label.set_color('white')
        ax5.xaxis.set_label_coords(.5, -0.09)
        ax5.yaxis.set_label_coords(-0.12, 0.5)
        ax5.set_ylabel('Source', fontsize = 4)
        ax5.yaxis.label.set_color('white')

        ax5.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax5.tick_params(axis='y', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax5.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)
        ax5.set_yticklabels(defineticks(20,5,-5),fontsize = 3.5)
        ax5.set_yticks(defineticks(4.5,19.5,5)) 
        ax5.set_xticks(defineticks(4.5,24.5,5))

        ax5.spines['bottom'].set_color('white')
        ax5.spines['top'].set_color('white')
        ax5.spines['left'].set_color('white')
        ax5.spines['right'].set_color('white')
        

        ## Plot 6: std(Y) WL 2
        sdFull=np.zeros((Ns,Nd)).flatten() 
        sdFull[Ia.flatten()]=stdY[Ib.flatten()+Nm]
        sdFull = np.reshape(sdFull,(Ns,Nd), order = 'F')

        ax6 = fig.add_axes([0.475, 0.4, 0.22, 0.22]) 
        im6 = ax6.imshow(sdFull, cmap = jet, extent = [-0.50, 28, -0.50, 24], aspect = 28/24, vmin = 0, vmax = 0.2)
        #p0=plt.gca()
        ax6.set_title('\u03C3 (Y) '+ str(wls[1]) + ' nm',color = 'w', y = 0.95)
        ax6.set_xlabel('Detector')
        ax6.xaxis.label.set_color('white')
        ax6.xaxis.set_label_coords(.5, -0.075)
        ax6.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax6.set_yticklabels(defineticks(5,25,5,blankLabels = True))
        ax6.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)
        ax6.set_xticks(defineticks(4.5,24.5,5))

        ax6.spines['bottom'].set_color('white')
        ax6.spines['top'].set_color('white')
        ax6.spines['left'].set_color('white')
        ax6.spines['right'].set_color('white')
        

        axins2 = inset_axes(ax6,
                        width="5%",  # width = 5% of parent_bbox width
                        height="80%",  # height : 80%
                        loc='lower right',
                        bbox_to_anchor=(0.12, 0.05, 1, 1),
                        bbox_transform=ax6.transAxes)
        cb2 = fig.colorbar(im6, cax=axins2, orientation="vertical",drawedges=False)
        cb2.set_ticks([0,0.075, 0.1, 0.2])
        cb2.set_ticklabels(['0','0.075','\u03C3 (Y)','0.2'])
        cb2.ax.yaxis.set_tick_params(color="white", pad = 0.8, length = 0)
        
        plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color="white")
        cb2.outline.set_edgecolor('white')
        cb2.outline.set_linewidth(0.5)


    #     # Plot 7: SNR WL 1
        snrN=np.zeros(np.shape(sdFull)).flatten()
        snrD=np.zeros(np.shape(sdFull)).flatten()
        snrN[Ia.flatten()]=Phi_0[Ib.flatten()]
        snrD[Ia.flatten()]=np.std(data[Ib.flatten()],1, ddof = 1)
        snrN = np.reshape(snrN,(Ns,Nd), order = 'F')
        snrD = np.reshape(snrD,(Ns,Nd), order = 'F')

        ax7 = fig.add_axes([0.315, 0.125, 0.22, 0.22]) 

        ax7.imshow(np.log10(snrN/snrD), cmap = jet, extent = [-0.50, 28, -0.50, 24], aspect = 28/24,vmin = 0.5, vmax = 2)#[0.5,2])
        ax7.set_title('SNR ' + str(wls[0]) + ' nm', color ='w', pad = -0.8)
        ax7.set_xlabel('Detector', fontsize = 4)
        ax7.set_ylabel('Source', fontsize = 4)
        ax7.yaxis.label.set_color('white')
        ax7.xaxis.label.set_color('white')
        ax7.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax7.tick_params(axis='y', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax7.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)
        ax7.set_yticklabels(defineticks(20,5,-5),fontsize = 3.5)
        ax7.set_yticks(defineticks(4.5,19.5,5)) 
        ax7.set_xticks(defineticks(4.5,24.5,5))
        ax7.xaxis.set_label_coords(.5, -0.09)
        ax7.yaxis.set_label_coords(-0.12, 0.5)

        ax7.spines['bottom'].set_color('white')
        ax7.spines['top'].set_color('white')
        ax7.spines['left'].set_color('white')
        ax7.spines['right'].set_color('white')
        


    #    ## Plot 8: SNR WL 2
        snrN = snrN.flatten()
        snrD = snrD.flatten()
        snrN[Ia.flatten()]=Phi_0[Ib.flatten()+Nm]
        snrD[Ia.flatten()]=np.std(data[Ib.flatten()+Nm],1,ddof = 1)
        snrN = np.reshape(snrN,(Ns,Nd), order = 'F')
        snrD = np.reshape(snrD,(Ns,Nd), order = 'F')
        ax8 = fig.add_axes([0.475, 0.125, 0.22, 0.22]) 
        im8 = ax8.imshow(np.log10(snrN/snrD), cmap = jet, extent = [-0.50, 28, -0.50, 24], aspect = 28/24,vmin = 0.5, vmax = 2)
        #p0=plt.gca()
        ax8.set_title('SNR ' + str(wls[1]) + ' nm',color = 'w', pad = -0.8)
        ax8.set_xlabel('Detector')
        ax8.xaxis.label.set_color('white')
        ax8.xaxis.set_label_coords(.5, -0.075)
        ax8.tick_params(axis='x', colors='white', pad = 1.5, length = 1, width = 0.5, direction = 'in')
        ax8.set_yticklabels(defineticks(5,25,5,blankLabels = True))
        ax8.set_xticklabels(defineticks(5,25,5), fontsize = 3.5)
        ax8.set_xticks(defineticks(4.5,24.5,5))

        ax8.spines['bottom'].set_color('white')
        ax8.spines['top'].set_color('white')
        ax8.spines['left'].set_color('white')
        ax8.spines['right'].set_color('white')
        
        axins3 = inset_axes(ax8,
                        width="5%",  # width = 5% of parent_bbox width
                        height="80%",  # height : 80%
                        loc='lower right',
                        bbox_to_anchor=(0.12, 0.05, 1, 1),
                        bbox_transform=ax8.transAxes)
        
        cb3 = fig.colorbar(im8, cax=axins3, orientation="vertical", drawedges=False)
        cb3.ax.yaxis.set_tick_params(color="white", pad = 0.8, length = 0)
        cb3.set_ticks([0.5, 1.2, 2])
        cb3.set_ticklabels(['1.2','SNR','10e2'])
        plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color="white")
        cb3.outline.set_edgecolor('white')
        cb3.outline.set_linewidth(0.5)

        ##Plot 10: Noise Histogram    
        keep=np.where(info["pairs"]["lambda"] == np.min(wls),1,0)
        dList=nm.repmat(range(1,Nd+1),Ns,1)
        measFull=np.c_[nm.repmat(range(1,Ns+1),Nd,1).flatten(),dList.flatten('F')]

        Ib = np.argwhere(np.isin(measFull, [np.c_[info["pairs"]["Src"][keep==1],info["pairs"]["Det"][keep==1]]]))
        
        #Ib0 = np.argwhere(Ib == 0)  #this check for zeros wasn't necessary since there weren't any; may cause issues with later functions 
        #Ib[Ib == 0] = []

        subkeep = np.argwhere(np.logical_and(np.where(info["pairs"]["r3d"] >= np.min(params["rlimits"]),1,0), np.where(info["pairs"]["r3d"] <= np.max(params["rlimits"]),1,0)))
        keep = np.intersect1d(subkeep, Ib)
        ax10.hist(stdY[keep]*100,np.arange(0,101,0.5),color = 'blue', ec = 'black', label = wavelengths[0])
        ax10.hist(stdY[keep+Nm]*100,np.arange(0,101,0.5), color ='lime', ec = 'black', label = wavelengths[1])
        yl=ax10.get_ylim()
        ax10.set_xlim([0,30])
        ax10.plot(np.ones([2,1])*params["bthresh"]*100,[0,yl[-1]],'r',linewidth = 2, label = 'Threshold')
        ax10.set_xlabel('\u03C3(y) [ % ]')
        ax10.set_ylabel('Measurements')
        ax10.yaxis.set_label_coords(-0.12, 0.5)
        ax10.set_title('Rsd from ' + str(np.min(params["rlimits"])) + ' - ' + str(np.max(params["rlimits"])) +' mm',color = 'white')
        ax10.set_facecolor('black')
        ax10.xaxis.label.set_color('white')
        ax10.yaxis.label.set_color('white')
        ax10.tick_params(axis='x', colors='white', length = 1, width = 0.5, direction = 'in')
        ax10.tick_params(axis='y', colors='white',length = 1, width = 0.5, direction = 'in')
        ax10.spines['bottom'].set_color('white')
        ax10.spines['top'].set_color('white')
        ax10.spines['left'].set_color('white')
        ax10.spines['right'].set_color('white')
        ax10.margins(y = 0)
        ax10.margins(x = 0)
        ax10.legend()



    def Plot_RawData_Metrics_II_DQC(data,info,params=None):
        """ 
        This function generates a single-page report that includes:
        zoomed raw time trace
        light fall off as a function of Rsd
        Power spectra for 830nm at 2 Rsd
        Histogram for measurement noise
        """
    # %% Parameters and Initialization
        if params == None:
            params = {}

        paramskeys = params.keys() 
        if "bthresh" not in paramskeys: 
            params["bthresh"]=0.075
            
        paramsrlimts = [[1,20],[21,35],[36,43]]

        if "rlimits" not in paramskeys: 
            params["rlimits"]=paramsrlimts
        elif len(params["rlimits"]) == 2: 
            concat = params["rlimits"] + paramsrlimts[1:]
            params["rlimits"] = concat
        elif len(params["rlimits"]) == 4: 
            params["rlimits"] = params["rlimits"] + paramsrlimts[2:]
        if "logfft" not in paramskeys: 
            params["logfft"]=0
            
            
        
        infosystemkeys = info['system'].keys() 
        if "init_framerate" in infosystemkeys: 
            fr=info['system']['init_framerate']
        else:
            fr=info['system']['framerate']
    
        wls=np.unique(info['pairs']['lambda'])
        Nwls=len(wls);
        wavelengths=[0]*Nwls
        for i in range(Nwls):
            wavelengths[i]=str(wls[i]) + ' nm'

        Ns=np.amax(info['pairs']['Src'])
        Nd=np.amax(info['pairs']['Det'])
        [Nm,Nt]=np.shape(data)
        
        
        if not np.isreal(data).all():
            data=abs(data)
        #casting to an int but may not be correct for all cases 
        Nm=int(Nm/Nwls)

        if Nt<(60/fr):
            ti=1
            tf=Nt
        elif Nt:
            ti=round(Nt/2)-round(5*fr)
            if ti<1:
                ti=1
            tf=round(Nt/2)+round(5*fr)
            if tf>=Nt:
                tf=Nt

    #     %% Check for good measurements
        infokeys = info.keys()
        if 'MEAS' in infokeys: 
            infoMeaskeys = info['MEAS'].keys()
            if 'GI' not in infoMeaskeys: 
                
                ##NOTE FOR LOGMEAN THERE ARE 2 ouputs so you have to choose which ones!!! 
                
                info = anlys.FindGoodMeas(tx4m.logmean(data)[0], info, params['bthresh'])
        else: 
            ##NOTE FOR LOGMEAN THERE ARE 2 ouputs so you have to choose which ones!!! 
            info = anlys.FindGoodMeas(tx4m.logmean(data)[0], info, params['bthresh'])
                               
    #     %% Light level fall off
        Phi_0=np.ndarray.mean(data,1)
        M=np.ceil(np.amax(np.log10(data)))
        yM=10**M
        where = np.where(info['pairs']['lambda'] == wls[0])[0]
        r=info['pairs']['r3d'][where]
        col = len(Phi_0)//Nm
        
        
        #figure settings 
        fig, fig_axes = plt.subplots(ncols= 2, nrows=2, figsize=(12,12), dpi=75,gridspec_kw={'height_ratios': [2, 1]}) #define figure once
        
        fig.set_facecolor('black')
        
        
        
        
    # %% Time trace bit


        where1 = np.where(info['pairs']['r3d'] >= np.min(params['rlimits']), 1,0)
        where2 = np.where(info['pairs']['r3d'] <= np.max(params['rlimits']), 1, 0)
        subwhere_3 = np.where(info['pairs']['lambda'] == np.max(wls), 1, 0)
        where3 = np.logical_and(info['MEAS']['GI'],  subwhere_3)    
        
        keep = np.logical_and(where1, where2)
        keep = np.logical_and(keep, where3)    
        
        if sum(keep):
            x = np.arange(ti+1,tf+2)/fr      
            y = np.squeeze(data[keep,ti:tf+1])    
            #First plot    
            #plot data
            
            fig_axes[0,0].semilogy(x,y.transpose(), '-', linewidth=1)

            #axis limits 
            minlim = min(x)
            maxlim = max(x)
            fig_axes[0,0].set_xlim([minlim,maxlim])
            fig_axes[0,0].set_ylim([1e-2,1e-1])

            #ticks and grid lines 
            fig_axes[0,0].spines['bottom'].set_color('white')
            fig_axes[0,0].spines['top'].set_color('white')
            fig_axes[0,0].spines['left'].set_color('white')
            fig_axes[0,0].spines['right'].set_color('white')


            #how things look
            fig_axes[0,0].tick_params(axis='x', colors='white')
            fig_axes[0,0].tick_params(axis='y', colors='white')
            fig_axes[0,0].xaxis.label.set_color('white')
            fig_axes[0,0].yaxis.label.set_color('white')
            fig_axes[0,0].set_xticks(np.arange(math.floor(minlim), math.ceil(maxlim), 1))
            fig_axes[0,0].set_facecolor('black')

            #axis labels and titles
            fig_axes[0,0].set_xlabel('Time [sec]', color='white')
            fig_axes[0,0].set_title('\u03A6(t) ' + wavelengths[1] + 'GI: Rsd from ' + str(np.min(params['rlimits'])) + '-'+ str(np.max(params['rlimits'])) + ' mm', color='white' )
        
        
        
        #Second plot    
        #plot data
        array = np.reshape(Phi_0,(Nm, len(Phi_0)//Nm), order='F')
        fig_axes[0,1].semilogy(r, array, '.', markersize=2)
        
        #axis stuff
        #axis limits 
        fig_axes[0,1].set_xlim([0,100])
        fig_axes[0,1].set_ylim([1e-6,yM])
        
        #ticks and grid lines 
        fig_axes[0,1].grid(axis="x", linestyle='-', linewidth=0.25)
        fig_axes[0,1].grid(axis="y", linestyle='--', linewidth=0.25, which='both')
        fig_axes[0,1].spines['bottom'].set_color('white')
        fig_axes[0,1].spines['top'].set_color('white')
        fig_axes[0,1].spines['left'].set_color('white')
        fig_axes[0,1].spines['right'].set_color('white')

        
        #how things look
        fig_axes[0,1].tick_params(axis='x', colors='white')
        fig_axes[0,1].tick_params(axis='y', colors='white')
        fig_axes[0,1].xaxis.label.set_color('white')
        fig_axes[0,1].yaxis.label.set_color('white')
        fig_axes[0,1].set_xticks(np.arange(0, 101, 10))
        fig_axes[0,1].set_facecolor('black')
        
        #axis labels
        fig_axes[0,1].set_xlabel('Source-Detector Separation ( mm )', color='white')
        fig_axes[0,1].set_ylabel('$\u03A6_{0}$ ( \u03bcW )', color='white')
        
        #legend 
        fig_axes[0,1].legend(wavelengths)
        
        

        if "Clipped" in info["MEAS"]:
            keep = np.where(info["MEAS"]["Clipped"]  == 1)[0]
            plt.semilogy(info["pairs"]["r3d"][keep],Phi_0[keep],'x')
            fig_axes[0,1].legend(wavelengths, "Clipped")


        ## <fft> for max wavelength at 2 distances
        lmdata = tx4m.logmean(data)[0]
        keep = np.logical_and(np.logical_and(np.logical_and((info["pairs"]["lambda"]==np.max(wls)) , info["MEAS"]["GI"]) , info["pairs"]["r3d"] >= params["rlimits"][0][0]) , info["pairs"]["r3d"]<=params["rlimits"][0][1])
        r1=np.mean(info["pairs"]["r3d"][keep])
        #% WL2reg_a=mean(squeeze(lmdata(keep,:)),1);
        WL2reg_a=lmdata[keep]
        ftdomain, ftmag0,_,_ = tx4m.fft_tts(WL2reg_a,fr)
        ftmag=matlab.rms_py(ftmag0)
        fig_axes[1,0].semilogx(ftdomain,ftmag,'--', color = 'red', linewidth = 1)

        keep=np.logical_and(np.logical_and(np.logical_and((info["pairs"]["lambda"] == np.max(wls)) , info["MEAS"]["GI"]) , info["pairs"]["r3d"] >= params["rlimits"][1][0]) , info["pairs"]["r3d"] <= params["rlimits"][1][1])
        r2=np.mean(info["pairs"]["r3d"][keep])

        #% WL2reg_b=mean(squeeze(lmdata(keep,:)),1)
        WL2reg_b=lmdata[keep]
        ftdomain, ftmag0,_,_ = tx4m.fft_tts(WL2reg_b,fr)
        ftmag = matlab.rms_py(ftmag0)
        fig_axes[1,0].semilogx(ftdomain,ftmag, '-', color = 'magenta', linewidth = 1)   
        fig_axes[1,0].set_xlim([1e-3,fr/2])
        fig_axes[1,0].set_xlabel('Frequency [Hz]')
        fig_axes[1,0].xaxis.label.set_color('white')
        fig_axes[1,0].grid(axis="x", linestyle='--', linewidth=0.25, which='both')
        fig_axes[1,0].grid(axis="y", linestyle='-', linewidth=0.25)    
        fig_axes[1,0].set_facecolor('black')
        fig_axes[1,0].set_ylabel('|P1 [au]|')
        fig_axes[1,0].yaxis.label.set_color('white')
        fig_axes[1,0].tick_params(axis='x', colors='white')
        fig_axes[1,0].tick_params(axis='y', colors='white')
        fig_axes[1,0].spines['bottom'].set_color('white')
        fig_axes[1,0].spines['top'].set_color('white')
        fig_axes[1,0].spines['left'].set_color('white')
        fig_axes[1,0].spines['right'].set_color('white')
        fig_axes[1,0].margins(y = 0)
        fig_axes[1,0].legend(['\u03BC('+ str(np.max(wls))+' nm, ~'+ format(r1,'.4f')+' mm)', '\u03BC('+str(np.max(wls))+' nm, ~'+format(r2,'.4f')+' mm)'])
            

        ## Noise Histogram
        if 'synchpts' in info["paradigm"]:
            NsynchPts = len(info["paradigm"]["synchpts"])# % set timing of data
            if NsynchPts > 2:
                tF = info["paradigm"]["synchpts"][-1]
                t0 = info["paradigm"]["synchpts"][1]
            elif NsynchPts == 2:
                tF = info["paradigm"]["synchpts"][1]
                t0 = info["paradigm"]["synchpts"][0]
            else:
                tF = np.shape(data)[1]
                t0 = 0
            
            stdY=np.std(lmdata[:, t0:tF],1,ddof = 1)
        else:
            stdY=np.std(lmdata,1,ddof = 1)
        
        keep=np.where(info["pairs"]["lambda"] == np.min(wls),1,0)
        dList=nm.repmat(range(1,Nd+1),Ns,1)
        measFull=np.c_[nm.repmat(range(1,Ns+1),Nd,1).flatten(),dList.flatten('F')]
        keep_srcs = info["pairs"]["Src"][keep==1]
        keep_dets = info["pairs"]["Det"][keep==1]
        cat_array = np.c_[keep_srcs, keep_dets]
     
        Ib = np.unique(np.where(np.isin(np.c_[info["pairs"]["Src"][keep==1],info["pairs"]["Det"][keep==1]], measFull[None,:]))[0])

        subkeep = np.argwhere(np.logical_and(np.where(info["pairs"]["r3d"] >= np.min(params["rlimits"]),1,0), np.where(info["pairs"]["r3d"] <= np.max(params["rlimits"]),1,0)))
        keep = np.intersect1d(subkeep, Ib)
        fig_axes[1,1].hist(stdY[keep]*100,np.arange(0,101,0.5),color = 'blue', ec = 'black', label = wavelengths[0])
        fig_axes[1,1].hist(stdY[keep+Nm]*100,np.arange(0,101,0.5), color ='lime', ec = 'black', label = wavelengths[1])
        yl=fig_axes[1,1].get_ylim()
        fig_axes[1,1].set_xlim([0,30])
        fig_axes[1,1].plot(np.ones([2,1])*params["bthresh"]*100,[0,yl[-1]],'r',linewidth = 2, label = 'Threshold')
        fig_axes[1,1].set_xlabel('\u03C3(y) [ % ]')
        fig_axes[1,1].set_ylabel('Measurements')
        fig_axes[1,1].set_title('Rsd from ' + str(np.min(params["rlimits"])) + ' - ' + str(np.max(params["rlimits"])) +' mm',color = 'white')
        fig_axes[1,1].set_facecolor('black')
        fig_axes[1,1].xaxis.label.set_color('white')
        fig_axes[1,1].yaxis.label.set_color('white')
        fig_axes[1,1].tick_params(axis='x', colors='white')
        fig_axes[1,1].tick_params(axis='y', colors='white')
        fig_axes[1,1].spines['bottom'].set_color('white')
        fig_axes[1,1].spines['top'].set_color('white')
        fig_axes[1,1].spines['left'].set_color('white')
        fig_axes[1,1].spines['right'].set_color('white')
        fig_axes[1,1].margins(y = 0)
        fig_axes[1,1].margins(x = 0)
        fig_axes[1,1].legend()

        
    
    def Plot_RawData_Time_Traces_Overview(data,info_in,params = None):
        """ 
        This function generates a plot of raw data time traces separated by
        wavelength (columns). The top row shows time traces for all measurements
        within a source-detector distance range (defaults as 0.1 - 5.0 cm). The
        bottom row shows the same measurements but including only measurements
        passing a variance threshold (default: 0.075) as well as vertical lines
        corresponding to the stimulus paradigm.
        """
        if params == None:
            params = {}

    # Parameters and Initialization
        Nwl=len(np.unique(info_in['pairs']['WL']))
        [Nm,Nt]=np.shape(data)
        if not data.any():#(np.isreal(data)):
            data=abs(data)
        if 'init_framerate' in info_in['system']: 
            fr=info_in['system']['init_framerate']
        else:
            fr=info_in['system']['framerate']

            
        dt=1/fr;                          
        t=np.linspace(1,Nt,Nt)*dt

        if params is None:
            params = {}
        if 'bthresh' not in params:
            params['bthresh'] = 0.075
        if 'rlimits' not in params:
            params['rlimits'] = [1,40]
        if 'yscale' not in params:
            params['yscale'] = 'log'

        lambdas, ind1 = np.unique(info_in['pairs']['lambda'],return_index=True)
        lambdas = lambdas[np.argsort(ind1)]

        WLs, ind2 = np.unique(info_in['pairs']['WL'],return_index=True)
        WLs = WLs[np.argsort(ind2)]


        ## Check for good measurements
        if ('MEAS' not in info_in) or ('GI' not in info_in['MEAS']):
            lmdata = tx4m.logmean(data)[0]
            info_out = anlys.FindGoodMeas(lmdata, info_in, params['bthresh'])
        else:
            info_out = info_in.copy()

        fig, fig_axes = plt.subplots(ncols= Nwl, nrows=2, dpi = 100, tight_layout =True) #define figure once
        fig.set_facecolor('black')

        ## Top row: all measurements broken apart by wavelength
        for j in range(0,Nwl):
            keep = np.logical_and((info_out['pairs']['r2d'] >= params['rlimits'][0]), (info_out['pairs']['r2d'] <= params['rlimits'][1]), (info_out['pairs']['WL'] == j))
            viz.PlotTimeTraceData(data[keep,:], t, params, fig_axes,[0,j])        
            fig_axes[0,j].set_xlim([0,max(t)+1])
            fig_axes[0,j].set_xlabel('Time [sec]')
            fig_axes[0,j].set_ylabel('\u03A6')
            if 'lambda' in info_out['pairs']: # this checks if lambda is a column in the table pairs, so we need to check if it is a field of the dict pairs
                fig_axes[0,j].set_title('All '+str(lambdas[j])+' nm, Rsd:'+ str(params['rlimits'][0])+'-'+ str(params['rlimits'][1])+' mm', color = 'white')
            else:
                fig_axes[0,j].set_title('All WL ## '+str(WLs[j])+' nm, Rsd:'+ str(params['rlimits'][0])+'-'+ str(params['rlimits'][1])+' mm', color = 'white')
            
                


        ## Bottom row: good measurements broken apart by wavelength
        for j in range(0,Nwl):
        # if 'MEAS' in info.keys(): # shouldn't need this because FindGoodMeas should create a field called 'MEAS' and 'GI'
            keep=np.logical_and(np.logical_and((info_out['pairs']['r2d'] >= params['rlimits'][0]), (info_out['pairs']['r2d'] <= params['rlimits'][1]), (info_out['pairs']['WL'] == j)), (info_out['MEAS']['GI']) ) #requires FindGoodMeas
            viz.PlotTimeTraceData(data[keep,:], t, params, fig_axes,[1,j])
            fig_axes[1,j].set_xlim([0,max(t)+1])
            fig_axes[1,j].set_xlabel('Time [sec]')
            fig_axes[1,j].set_ylabel('\u03A6',)
            if 'lambda' in info_out['pairs']:
                fig_axes[1,j].set_title('Good '+str(lambdas[j])+' nm, Rsd:'+ str(params['rlimits'][0])+'-'+ str(params['rlimits'][1])+' mm',color = 'white')
            else:
                fig_axes[1,j].set_title('Good WL ## '+str(WLs[j])+' nm, Rsd:'+ str(params['rlimits'][0])+'-'+ str(params['rlimits'][1])+ ' mm', color = 'white')
            if 'paradigm' in info_out: # Add in experimental paradigm timing
               viz.DrawColoredSynchPoints(info_out,fig_axes[1,j],1)

    
    def cylinder(r, h = 1, a =0, nt=100, nv =50):
        """
        parameterize the cylinder of radius r, height h, base point a
        
        See: PlotCapData
        """
        theta = np.linspace(0, 2*np.pi, nt)
        v = np.linspace(a, a+h, nv )
        theta, v = np.meshgrid(theta, v)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = v
        return x, y, z


    def PlotCapData(SrcRGB, DetRGB, info, fig_axes = None,params = None):
        """
        PLOTCAPDATA A basic plotting function for generating and labeling cap grids.

        PLOTCAPDATA(SrcRGB, DetRGB, info) plots the input RGB information in
        one of three modes:

        'text' - Optode numbers are arranged in a cap grid and colored with the
        RGB input.
        'patch' - Optodes are plotted as patches and colored with the RGB
        input.
        'textpatch' - Optodes are plotted as patches and colored with the RGB
        input, with optode numbers overlain in white.

        PLOTCAPDATA(SrcRGB, DetRGB, info, params) allows the user to specify
        parameters for plot creation.

        "params" fields that apply to this function (and their defaults):
            fig_size    [20, 200, 1240, 420]    Default figure position vector.
            fig_handle  (none)                  Specifies a figure to target.
                                                If empty, spawns a new figure.
            dimension   '2D'                    Specifies either a 2D or 3D
                                                plot rendering.
            mode        'textpatch'             Display mode.

        See Also: PLOTCAP, PLOTCAPGOODMEAS, PLOTCAPMEANLL.
        """
        if params == None:
            params = {}
        ## Parameters and Initialization.
        # Nm = height(info.pairs);
        Ns = len(info['optodes']['spos2'])
        Nd = len(info['optodes']['dpos2'])

        Srcs = np.arange(1,Ns+1) #%unique(info.pairs.Src);
        Dets = np.arange(1,Nd+1) #%unique(info.pairs.Det);

        BkgdColor = [0, 0, 0]; # KEEP THESE LIKE THIS!
        #% LineColor = [1, 1, 1] # They are read as RGB triplets by later functions.

        box_buffer = 5
        new_fig = 0

        if 'dimension' not in params or params['dimension'] == []: 
            params['dimension'] = '2D'

        if 'rhombus' not in params:
            params['rhombus'] = 1 
        if 'eeg_style' not in params:
            params['eeg_style'] = 0 
        if 'LineColor' not in params:
            params['LineColor'] = [1,1,1] 

        if params['dimension'] == '2D':
        #  Get optode positions.
            spos = info['optodes']['spos2']
            dpos = info['optodes']['dpos2']
            
            if params['rhombus']:
                # Calculate side length and create square vectors.
                l = la.norm(spos[0, :] - spos[1, :]) / 2
                xsq = [l, 0, -l, 0] #  Okay, they're really rhombi, but you get the point.
                ysq = [0, -l, 0, l]
            else:
                #  Calculate side length and create square vectors.
                nn1S=info['pairs']['Src'][np.where(info['pairs']['NN'] == 1)[0][0]] 
                nn1D=info['pairs']['Det'][np.where(info['pairs']['NN'] == 1)[0][0]]

                l = la.norm(spos[0, :] - dpos[0, :]) / 2
                xsq = [l, -l, -l, l] # Okay, they're really rhombi, but you get the point.
                ysq = [l, l, -l, -l]
            
            if params['eeg_style']:
                l = la.norm(spos[0, :] - spos[1, :]) / 2
                xsq,ysq,z = viz.cylinder(2)
                xsq=xsq[0,:]*(l/7)
                ysq=ysq[0,:]*(l/7)

        #   Default figure size.
            if 'fig_handle' not in params: 
                if 'fig_size' not in params or params['fig_size'] == []: 
                    params['fig_size'] = [20, 200, 1240, 420]

        if params['dimension'] == '3D':
        #     Get optode positions.
            spos = info['optodes']['spos3']
            dpos = info['optodes']['dpos3']
        
        #   [xdir, ydir, zdir] = CheckOrientation(info) #checkorientation may need to be written in Python later
            
    #          Default figure size.
            if 'fig_handle' not in params: 
                if 'fig_size' not in params or params['fig_size'] == []:
                    params['fig_size'] = [20, 200, 560, 560]
        if 'mode' not in params or not bool(params['mode']): 
            params['mode'] = 'textpatch'  

        if params['mode'] == 'good':
            params['mode'] = 'text'
        if params['mode'] == 'text':
            STextColor = SrcRGB
            DTextColor = DetRGB
            if params['dimension'] == '2D':
                TextSize =25
            if params['dimension'] == '3D':
                TextSize = 15
        if params['mode'] == 'patch':
            if params['dimension'] == '2D':
    #           % Nothing for now.
                pass
            if params['dimension'] == '3D':
                MarkerSize = 6
                SMarkerEdgeColor = SrcRGB
                DMarkerEdgeColor = DetRGB
    #         
        if params['mode'] ==  'textpatch':
            # Smaller text and larger markers needed when plotting text over markers.
            if params['dimension'] =='2D':
                TextSize = 8
                STextColor = nm.repmat(params['LineColor'], Ns, 1)
                DTextColor = nm.repmat(params['LineColor'], Nd, 1)
            if params['dimension'] == '3D':
                TextSize = 6
                MarkerSize = 9
                STextColor = nm.repmat(params['LineColor'], Ns, 1)
                DTextColor = nm.repmat(params['LineColor'], Nd, 1)
                SMarkerEdgeColor = STextColor
                DMarkerEdgeColor = DTextColor
             
        # Draw data.
        if params['dimension'] =='2D':
            if isinstance(fig_axes, type(None)):
                print('creating figure because there is none')
                fig, fig_axes = plt.subplots(1,1,facecolor = 'black',dpi = 300) 
            if params['eeg_style']:
                # Reshape input before drawing - this is crucial for patch.
                SrcRGB = SrcRGB.reshape(-1,1,3)
                DetRGB = DetRGB.reshape(-1,1,3)

                # Srcs
                x =np.transpose((np.transpose(nm.repmat(spos[:, 0],1,np.shape(xsq)[0])) + nm.repmat(xsq, 1, Ns)))
                y = np.transpose((np.transpose(nm.repmat(spos[:, 1], np.shape(xsq)[0],1)) + nm.repmat(ysq, Ns, 1)))
                xy = np.stack((x,y), axis = 0)
                numSquares = len(x[0])
                for i in range(numSquares):
                    polygonxy = []
                    for j in range(len(x)):
                        sx = x[j][i]
                        sy = y[j][i]
                        polygonxy.append([sx,sy])
                    color = SrcRGB[i-1,:,:][0]
                    square = plt.Polygon(polygonxy,color = color)
                    fig_axes.add_patch(square) ## 3/11/22
                
    #             % Dets
                x =np.transpose((np.transpose(nm.repmat(dpos[:, 0],np.shape(xsq)[0],1)) + nm.repmat(xsq, Nd, 1)))
                y = np.transpose((np.transpose(nm.repmat(dpos[:, 1], np.shape(xsq)[0],1)) + nm.repmat(ysq, Nd, 1)))
                numSquares = len(x[0])
                for i in range(numSquares): 
                    polygonxy = []
                    for j in range(len(x)):
                        sx = x[j][i]
                        sy = y[j][i]
                        polygonxy.append([sx,sy])
                    color = DetRGB[i,:,:][0]
                    square = plt.Polygon(polygonxy,color = color)
                    fig_axes.add_patch(square)## 3/11/22
                    fig_axes.set_ylim(np.min(y)-30,np.max(y))## 3/11/22
                    fig_axes.set_xlim(np.min(x)-20,np.max(x)+20)## 3/11/22
            
            elif params['mode'] == 'patch':
                # Reshape input before drawing - this is crucial for patch.
                SrcRGB = SrcRGB.reshape(-1,1,3)
                DetRGB = DetRGB.reshape(-1,1,3)

                #  # Srcs
                x =nm.repmat(spos[:, 0],4,1) + np.transpose(nm.repmat(xsq, Ns, 1))
                y = nm.repmat(spos[:, 1],4, 1) + np.transpose(nm.repmat(ysq, Ns, 1))
                xy = np.stack((x,y), axis = 0)
                numSquares = len(x[0])
                for i in range(numSquares):
                    polygonxy = []
                    for j in range(len(x)):
                        sx = x[j][i]
                        sy = y[j][i]
                        polygonxy.append([sx,sy])
                    color = SrcRGB[i,:,:][0]
                    square = plt.Polygon(polygonxy,color = color)
                    square.set_edgecolor(color)
                    square.set_linewidth(0.1)
                    fig_axes.add_patch(square)         ## 3/11/22   
                
    #             % Dets
                x = nm.repmat(dpos[:, 0],4,1) + np.transpose(nm.repmat(xsq, Nd, 1))
                y = nm.repmat(dpos[:, 1],4, 1) + np.transpose(nm.repmat(ysq, Nd, 1))
                xy = np.stack((x,y), axis = 0)

                numSquares = len(x[0])
                for i in range(numSquares): 
                    polygonxy = []
                    for j in range(len(x)):
                        sx = x[j][i]
                        sy = y[j][i]
                        polygonxy.append([sx,sy])
                    color = DetRGB[i,:,:][0]
                    square = plt.Polygon(polygonxy,color = color)
                    square.set_edgecolor(color)
                    square.set_linewidth(0.1)
                    fig_axes.add_patch(square)## 3/11/22
                    
                fig_axes.set_ylim(np.min(y)-30,np.max(y))## 3/11/22
                fig_axes.set_xlim(np.min(x)-20,np.max(x)+20)## 3/11/22
            if params['mode'] == 'good':
                params['mode'] = 'text'
            if params['mode'] == 'text':
                for s in range(0,Ns):
                    ## 3/11/22
                    fig_axes.text(spos[s, 0], spos[s, 1], str(Srcs[s]), color =  STextColor[s, :], verticalalignment = 'center', horizontalalignment = 'center', fontsize = TextSize, fontweight = 'bold')
                for d in range(0,Nd):
                    ## 3/11/22
                    fig_axes.text(dpos[d, 0], dpos[d, 1], str(Dets[d]), color = DTextColor[d, :], verticalalignment = 'center',horizontalalignment = 'center', fontsize = TextSize, fontweight = 'bold')
                xsmax = np.amax(spos[:,0])
                xsmin = np.amin(spos[:,0])
                xdmax = np.amax(dpos[:,0])
                xdmin = np.amin(dpos[:,0])
        
        if params['dimension'] =='3D':  #case '3D'
            if params['mode'] == 'patch':
                #Srcs
                # pylab.ion()
                fig = pylab.figure()
                ax3 = Axes3D(fig)
                for s in range(0,Ns):
                    ax3.scatter(spos[s,0],spos[s,2],spos[s,1],color = SrcRGB[s,:], marker = 'o', s = 500) # reoptimize with spheres at a later date
    #             % Dets
                for d in range(0, Nd):
                    ax3.scatter(dpos[d,0],dpos[d,2],dpos[d,1],color = DetRGB[d,:], marker = 'o', s =500)
                ax3.view_init(30,-150)

            
            if params['mode'] == 'text':
                for s in range(0,Ns):
                    plt.text(spos[s,0],spos[s,2], spos[s,1],str(Srcs[s]), color = STextColor[s,:], verticalalignment = 'center',horizontalalignment = 'center', fontsize = TextSize, fontweight = 'bold')
                for d in range(0,Nd):
                    plt.text(dpos[d,0],dpos[d,2], dpos[d,1],str(Dets[d]), color = DTextColor[d,:], verticalalignment = 'center',horizontalalignment = 'center', fontsize = TextSize, fontweight = 'bold')

        if params['dimension'] =='3D': 
            return ax3, x, y, True 
        else: 
            if params['mode'] == 'patch':
                return x, y, False, fig_axes
            else:
                return max(xsmax,xdmax), min(xsmin,xdmin), fig_axes

    
    def PlotCapGoodMeas(info, fig_axes = None,params = None):
        """ 
        PLOTCAPGOODMEAS A Good Measurements visualization overlaid on a cap grid.

        PLOTCAPGOODMEAS(info) plots a visualization of the Good Measurements
        determined by FINDGOODMEAS and arranges them based on the metadata in
        "info.optodes". Good channels are depicted as green lines, bad channels
        red lines; sources and detectors are given lettering in light blue and
        red.

        The plot title provides tallies for all specified groupings. The next
        line of the title lists how many optodes for which only 33% of their
        measurements are good. These optodes are surrounded with white circles.

        PLOTCAPGOODMEAS(info, params) allows the user to specify parameters for
        plot creation.

        "params" fields that apply to this function (and their defaults):
            fig_size    [20, 200, 1240, 420]    Default figure position vector.
            fig_handle  (none)                  Specifies a figure to target.
                                                If empty, spawns a new figure.
            dimension   '2D'                    Specifies either a 2D or 3D
                                                plot rendering.
            rlimits     (all R2D)               Limits of pair radii displayed.
            Nnns        (all NNs)               Number of NNs displayed.
            Nwls        (all WLs)               Number of WLs averaged and
                                                displayed.
            mode        'good'                  Display mode. 'good' displays
                                                channels above noise threhsold,
                                                'bad' below.

        Dependencies: PLOTCAPDATA, ISTABLEVAR.
        
        See Also: FINDGOODMEAS, PLOTCAP, PLOTCAPMEANLL.
        """
        # %% Paramters and Initialization.
        if params == None:
            params = {}
            
        BkgdColor = 'k'
        LineColor = 'w'
        SMarkerColor = [1, .75, .75]
        DMarkerColor = [.55, .55, 1]
        use_NNx_RxD = 'RxD'
        Nm = len(info['pairs']['Src'])
        Ns = len(np.unique(info['pairs']['Src']))
        Nd = len(np.unique(info['pairs']['Det']))
        cs = np.unique(info['pairs']['WL'])
        Nc = len(cs)


        thr = 0.33
        N_bad_SD = 0 #% Number above threshold.
        thr_NNx_RxD = 33

        if not 'mode' in params or len(params['mode']) == 0:
            params['mode'] = 'good'
        
        if not bool(params['useGM']) or 'MEAS' not in info or ('MEAS' in info and 'GI' not in info['MEAS']): 
            GM = np.ones((Nm, 1))
        else:
            GM = info['MEAS']['GI']
        
        if 'dimension' not in params or params['dimension'] == 0:
            params['dimension'] = '2D'


        if ('rlimits' not in params or len(params['rlimits']) == 0 ) and ('Nnns' not in params or len(params['Nnns'])== 0): 
        # If both empty, use ALL.
            use_NNx_RxD = 'all'
            params['rlimits'] = [np.min(info['pairs']['r' + str(np.char.lower(params['dimension']))]),np.max(info['pairs']['r' + str(np.char.lower(params['dimension']))])]
            lvar = 1
            
        else: # % Otherwise, set defaults if one or the other is missing.
            if 'rlimits' not in params or len(params['rlimits']) == 0 :
                use_NNx_RxD = 'NNx'
                lvar = params['Nnns']
                thr_NNx_RxD = 2

            if 'Nnns' not in params or len(params['Nnns'])== 0 : 
                lvar = [i for i in range(1,len(params['rlimits']))]
        
        if params['dimension'] =='2D':
            Ndim = 2
            spos = info['optodes']['spos2']
            dpos = info['optodes']['dpos2']
            MarkerSize = 20
            if 'fig_size' not in params   or  len(params['fig_size']) == 0:
                params['fig_size'] = [20, 200, 1240, 420]

        if params['dimension'] =='3D':
            Ndim = 3
            spos = info['optodes']['spos3']
            dpos = info['optodes']['dpos3']
            MarkerSize = 20
            if 'fig_size' not in params   or  len(params['fig_size']) == 0:
                params['fig_size'] = [20, 200, 560, 560]

        if params['mode'] in ['good','both']:
            modeGM = GM
            SDLineColor = 'g'
            l_Thickness = 0.5 * np.ones((len(lvar), 1))
        if params['mode'] == 'bad':
            modeGM = np.logical_not(GM)
            SDLineColor = 'r'
            l_Thickness = np.arange(0.5 + 1.5 * len(lvar), 0.5-1, -1.5)
            
        # %% Plot Src and Det #s.
        params2 = params.copy()
        params2['mode'] = 'text'
        SrcRGB = nm.repmat(SMarkerColor, Ns, 1)
        DetRGB = nm.repmat(DMarkerColor, Nd, 1)


        if params['mode'] == 'good' or params['mode'] == 'bad':
            xmax,xmin, fig_axes =viz.PlotCapData(SrcRGB, DetRGB, info, fig_axes, params2)
            title = str(np.char.upper(params['mode'][0])) +  str(params['mode'][1:]) + ' Measurements\n'
            fig_axes.set_facecolor('black')

        if params['mode'] == 'bad':
            title =  'Both Good and Bad Measurements\n'
            params['mode']='bad'
            viz.PlotCapGoodMeas(info, params)
        # %% Plot GM Lines.
        for k in lvar:
            # % Ignore WLs.
            if use_NNx_RxD == 'RxD':
                    infopairs_dimension = info['pairs']['r'+ str(np.char.lower(params['dimension']))]

                    if len(lvar) == 1: 
                        idxGreaterThan = np.where(infopairs_dimension >= params['rlimits'][0], 1 , 0)
                        idxLessThan = np.where(infopairs_dimension <= params['rlimits'][1],  1 , 0)
                    else: 
                        idxGreaterThan = np.where(infopairs_dimension >= params['rlimits'][k-1, 0], 1 , 0)
                        idxLessThan = np.where(infopairs_dimension <= params['rlimits'][k-1, 1],  1 , 0)


                    keep_NNx_RxD = np.logical_and(idxGreaterThan, idxLessThan)
        
            
            elif use_NNx_RxD == 'NNx':   
                keep_NNx_RxD = np.where(info['pairs']['NN'] == k, 1,0)

            elif use_NNx_RxD == 'all':
                keep_NNx_RxD = np.ones((Nm, 1));
            
            keep_NNx_RxD = np.where(keep_NNx_RxD == True, 1, 0)
            modeGM = np.where(modeGM == True, 1, 0).flatten()
            keep = np.logical_and(keep_NNx_RxD, np.transpose(modeGM))
            keep_idx = np.where(keep>0)[0] #% NEED THIS FOR THE NEXT SECTION
    #         % This reshapes position vectors of S-D pairs so "plot" will plot each
    #     % pair as individual line, by inserting NaN between each pair.
        
            array1 = np.array(np.transpose(spos[info['pairs']['Src'][keep_idx]-1, :].flatten('F') ))
            array2 = np.array(np.transpose(dpos[info['pairs']['Det'][keep_idx]-1, :].flatten('F')))
            array3 = np.empty((1, (len(keep_idx)) * Ndim))
            array3[:] = np.NaN

            array = np.vstack((array1,array2,array3))
            array = array.flatten('F')
            pos = np.reshape(array, (-1,Ndim), order = 'F')
            if params['dimension'] == '2D':
                fig_axes.plot(pos[:,0],pos[:,1], color = SDLineColor,linewidth = 2)
            if params['dimension'] == '3D':
                fig = pylab.figure()
                ax3 = Axes3D(fig)
                ax3.scatter(pos[:,0],pos[:,1], pos[:,2], LineColor = SDLineColor,linewidth = 1.5)

            N_GMs = np.zeros((lvar)) 
            N_Tots = np.zeros((lvar))

            N_GMs[k-1] = np.count_nonzero(keep)
            N_Tots[k-1] = np.count_nonzero(keep_NNx_RxD)        

        #     %% Calculate and Mark Any Srcs and Dets Above Threshold.
        # % IE, for each Src or Det, we count it as GM if >1 of its WLx is GM. Since
        # % each Src/Det will have WLx * Det # of channels, we can simply count the
        # % number of unique Det/Src for each Src/Det's GMs.
        if use_NNx_RxD == 'RxD' or use_NNx_RxD == 'all':
            keep_NNx_RxD = np.where(info['pairs']['r'+str(np.char.lower((params['dimension'])))] <= thr_NNx_RxD, 1,0)
        if use_NNx_RxD == 'NNx':
            keep_NNx_RxD = np.where(info['pairs']['NN'] <= thr_NNx_RxD, 1, 0) 
        for s in range(1, Ns+1):
            where1 = np.where(info['pairs']['Src']==s, 1,0) 
            keep = np.logical_and(np.logical_and(where1, keep_NNx_RxD), GM) # this logical and isn't working,  in updated validation script it works though
            N_GM_Src = np.unique(info['pairs']['Det'][keep]).size #

        #     % Total number of meas's for this Src.
            tot_Srcs = np.unique(info['pairs']['Det'][np.logical_and(np.where(info['pairs']['Src']==s,1,0) , keep_NNx_RxD)]).size #total srcs is being done correctly
            if (N_GM_Src/tot_Srcs) < (1 - thr):
                if params['dimension'] =='2D':
                    fig_axes.scatter(spos[s-1,0], spos[s-1,1], marker = 'o', facecolors = 'none', edgecolors = 'w', s = 3000)
                if params['dimension'] == '3D':
                    ax3.scatter(spos[s-1,0], spos[s-1,1], spos[s-1,2], marker = 'o', facecolors = 'none', edgecolors = 'w')

                N_bad_SD = N_bad_SD + 1


        for d in range(1, Nd +1):
            where1 = np.where(info['pairs']['Det']==d, 1,0) 
            keep = np.logical_and(np.logical_and(where1 , keep_NNx_RxD), GM)
            # keep = np.where(keep == 1)
            N_GM_Det = np.unique(info['pairs']['Src'][keep]).size
            tot_Dets = np.unique(info['pairs']['Src'][np.logical_and((info['pairs']['Det']==d) , keep_NNx_RxD)]).size
            if (N_GM_Det/tot_Dets) < (1 - thr):
                if params['dimension'] =='2D':
                    fig_axes.scatter(dpos[d-1,0], dpos[d-1,1], marker = 'o', facecolors = 'none', edgecolors = 'w', s = 3000)
                if params['dimension'] == '3D':
                    ax3.scatter(dpos[d-1,0], dpos[d-1,1], dpos[d-1,2],marker = 'o', facecolors = 'none', edgecolors = 'w')
                N_bad_SD = N_bad_SD + 1
        
        #     %% Add Title
        for k in lvar:
            title = title + str(int(N_GMs[0]))+'/'+str(int(N_Tots[0]))+ ' (' + "{:2.0f}".format(100*(N_GMs[0]/N_Tots[0]))+'%) '#str(100*(N_GMs[0]/N_Tots[0]))
            if use_NNx_RxD == 'NNx':
                title = title + 'NN' + str(k) +'\n'
            if use_NNx_RxD == 'RxD':
                title = title + 'r'+ str(np.char.lower(params['dimension'])) + ' \u220A [' + str(params['rlimits'][0]) + ', ' +str(params['rlimits'][1]) + '] mm\n'
            if k != lvar[-1]:
                title = title + ', ' 
        title = title + str(N_bad_SD) + ' Srcs or Dets have > ' + "{:2.0f}".format(100*thr) + '% of Bad Measurements'

        if use_NNx_RxD == 'NNx':
            title = title + ' \u2264 NN' + str(thr_NNx_RxD)
        if use_NNx_RxD == 'RxD':
            title = title + ', r' + str(np.char.lower(params['dimension'])) + ' \u2264 ' + str(thr_NNx_RxD)
        fig_axes.set_title(title, color = 'w', fontsize = 30)

        if xmax > 150:
            fig_axes.set_xlim(xmin, xmax+5) # was +/-50
        else:
            fig_axes.set_xlim(xmin-50, xmax+50)

    
    def PlotCapMeanLL(data, info, fig_axes = None, params = None):
        """ 
        PLOTCAPMEANLL A visualization of mean light levels overlaid on a cap
        grid.

        PLOTCAPMEANLL(data, info) plots an intensity map of the mean light
        levels for specified measurement groupings of each optode on the cap
        and arranges them based on the metadata in "info.optodes".

        PLOTCAPMEANLL(data, info, params) allows the user to specify parameters
        for plot creation.

        "params" fields that apply to this function (and their defaults):
            fig_size    [20, 200, 1240, 420]    Default figure position vector.
            fig_handle  (none)                  Specifies a figure to target.
                                                If empty, spawns a new figure.
            dimension   '2D'                    Specifies either a 2D or 3D
                                                plot rendering.
            rlimits     (all R2D)               Limits of pair radii displayed.
            Nnns        (all NNs)               Number of NNs displayed.
            Nwls        (all WLs)               Number of WLs averaged and
                                                displayed.
            useGM       0                       Use Good Measurements.
            Cmap.P      'hot'                   Default color mapping.

        Dependencies: PLOTCAPDATA, ISTABLEVAR, APPLYCMAP.

        See Also: PLOTCAP, PLOTCAPGOODMEAS.
        
        """
        if 'fig_axes' == None:
            fig, fig_axes = plt.figure(facecolor = 'black')

        if params == None:
            params = {}
        # %% Parameters and Initialization.
        LineColor = 'w'
        BkgdColor = 'k'
        Nm = int(data.shape[0])

        Ns = len(np.unique(info['pairs']['Src']))
        Nd = len(np.unique(info['pairs']['Det']))
        cs = np.unique(info['pairs']['WL']) # WLs.
        use_NNx_RxD = 'RxD'

        dims = data.shape
        Nt = int(dims[-1])
        NDtf = (len(dims) > 2)
        # Nm = np.prod(dims[0:-2], axis = 0) # this seems to be wrong and also seems duplicated
        keep_NNx_RxD = np.zeros((1, Nm))


        # Hardcoded params for color mapping to the units scaling used here.
        scale_order = 1e-2
        base = 1e9
        dr = 3
        params['mode'] = 'patch'
        if 'Scale' not in params:
            params['Scale'] = dr
        if 'abs' not in params:
            params['abs'] = 0 # Absolute or relative coloring
        params['PD'] = 1
        params['Th'] = {}
        params['Th']['P'] = 0
        params['DR'] = 1000

        if 'Cmap' not in params or not bool(params['Cmap']) or 'P' not in params['Cmap'] or params['Cmap']['P'] == 0:
            params['Cmap'] = {}
            params['Cmap']['P'] = 'hot'

        if 'dimension' not in params or not bool(params['dimension']): 
            params['dimension'] = '2D'
            
        if ('rlimits' not in params or params['rlimits'] == []) and ('Nnns' not in params or params['Nnns'] == []): 
        # If both empty, use ALL.
            use_NNx_RxD = 'all'
            lvar = 1
        else: # Otherwise, set defaults if one or the other is missing.
            if 'rlimits' not in params or params['rlimits'] == []: 
                use_NNx_RxD = 'NNx'
                lvar = params['Nnns']
            if 'Nnns' not in params or params['Nnns'] == []: 
                lvar = range(0, np.shape(params['rlimits'])[0])

        if 'Nwls' not in params or params['Nwls'] == []:
            params['Nwls'] = np.transpose(cs)
        
        if 'useGM' not in params or params['useGM'] == []: 
            params['useGM'] = 0
        
        if not params['useGM'] or 'MEAS' not in info or ('MEAS' in info and 'GI' in info['MEAS']): 
            GM = np.ones((Nm, 1))
        else:
            GM = info['MEAS']['GI']
        
        # if 'dimension' not in params or params['dimension'] ==[]:
        #     params['dimension'] = '2D' # '2D' | '3D'
        
        if 'fig_size' not in params or params['fig_size'] == []:
            if params['dimension'] == '2D':
                params['fig_size'] = [20, 200, 1240, 420]
            if params['dimension'] == '3D':
                params['fig_size'] = [20, 200, 560, 560]

        if 'lambda' in info['pairs']:
            lambdas = np.unique(info['pairs']['lambda'])
            if np.size(params['Nwls']) > 1:
                lambda_unit1 = '['
                lambda_unit2 = '] nm'
            else:
                lambda_unit1 = ''
                lambda_unit2 = ' nm'
        else:
            lambdas = cs
            lambda_unit1 = '\u03BB'
            lambda_unit2 = ''

        

        ##  %% N-D Input.
        if NDtf:
            data = np.reshape(data, [], Nt)

        ## Average LLs of each Src and Det.
        # Average LLs over time first.
        data = np.mean(data, 1)

        
        if use_NNx_RxD == 'RxD':
            for k in range(0, np.size(params['rlimits'], 0)):
                dimstr = 'r'+ str(np.char.lower(params['dimension']))
                infopairs_dimension = info['pairs'][dimstr]
                if k > 1:
                    idxGreaterThan = np.where(infopairs_dimension >= params['rlimits'][k-1][0], 1 , 0)
                    idxLessThan = np.where(infopairs_dimension <= params['rlimits'][k-1][1],  1 , 0)
                else:
                    idxGreaterThan = np.where(infopairs_dimension >= params['rlimits'][0], 1 , 0)
                    idxLessThan = np.where(infopairs_dimension <= params['rlimits'][1],  1 , 0)
                keep_NNx_RxD = np.logical_or(keep_NNx_RxD, np.logical_and(idxGreaterThan, idxLessThan))

        if use_NNx_RxD == 'NNx':        
            for k in params['Nnns']:
                idxEqualTo = np.where(info['pairs']['NN'] == k, 1 , 0)
                keep_NNx_RxD = np.logical_or(keep_NNx_RxD, idxEqualTo)
                unique, counts = np.unique(keep_NNx_RxD, return_counts=True)

            #IF it is a single number WE NEED BELOW 
            # k = params['Nnns']
            # idxEqualTo = np.where(info['pairs']['NN'] == k, 1 , 0)
            # keep_NNx_RxD = np.logical_or(keep_NNx_RxD, idxEqualTo)

        if use_NNx_RxD == 'all':
            keep_NNx_RxD = np.logical_not(keep_NNx_RxD)

        # % Desired WLs.
        keepWL = np.zeros((1, Nm))

        for k in params['Nwls']:
            idxEqualTo = np.where(info['pairs']['WL'] == k, 1 , 0)
            keepWL = np.logical_or(keepWL, idxEqualTo)

        if params['abs']:
            # % Src averages. (in desired R)
            Srcval = np.zeros(Ns)
            GM_Transposed = np.transpose(GM)
            for s in range(Ns):
                idxEqualTo = np.where(info['pairs']['Src'] == s+1, 1 , 0)
                keep = np.logical_and(idxEqualTo, np.logical_and(keep_NNx_RxD, np.logical_and(keepWL, GM_Transposed)))[0]
                Srcval[s] = np.log10(base * np.mean(data[keep]))

            
        #     % Det averages. (in desired R)
            Detval = np.zeros(Nd)
            for d in range(Nd):
                idxEqualTo = np.where(info['pairs']['Det'] == d+1, 1 , 0)
                keep = np.logical_and(idxEqualTo, np.logical_and(keep_NNx_RxD, np.logical_and(keepWL, GM_Transposed)))[0]
                Detval[d] = np.log10(base * np.mean(data[keep]));
            
        else:
            Srcval = np.zeros(Ns)
            GM_Transposed = np.transpose(GM)
        #     % Src averages. (in desired R)
            for s in range(Ns):
                idxEqualTo = np.where(info['pairs']['Src'] == s+1, 1 , 0)
                keep = np.logical_and(idxEqualTo, np.logical_and(keep_NNx_RxD, np.logical_and(keepWL, GM_Transposed)))[0]
                Srcval[s] = np.log10(np.mean(data[keep]))
            
        #     % Det averages. (in desired R)
            Detval = np.zeros(Nd)
            for d in range(Nd):
                idxEqualTo = np.where(info['pairs']['Det'] == d+1, 1 , 0)
                keep = np.logical_and(idxEqualTo, np.logical_and(keep_NNx_RxD, np.logical_and(keepWL, GM_Transposed)))[0]
                Detval[d] = np.log10(np.mean(data[keep]))


        # %% Scaling and Color Mapping
        if params['abs']: # % Absolute data values color mapping
            Srcval = Srcval - (np.log10(base) - dr)
            Detval = Detval - (np.log10(base) - dr)
        else:        #% Relative data values
            M=max([np.amax(Srcval),np.amax(Detval)])
            Srcval = Srcval - (M - dr)
            Detval = Detval - (M - dr)    

        stacked = np.hstack([Srcval,Detval])
        SDRGB, CMAP, params = viz.applycmap(stacked, [], params)
        

        SrcRGB = SDRGB[0:Ns, :]
        DetRGB = SDRGB[Ns:, :] 

        #  Send Light Levels to PlotCapData.
        if fig_axes == None:
            x,y, threeDGraph,fig_axes = viz.PlotCapData(SrcRGB, DetRGB, info, fig_axes,params)
            # fig.patch.set_facecolor('black')

        else:
            x,y, threeDGraph,fig_axes = viz.PlotCapData(SrcRGB, DetRGB, info, fig_axes, params)
            
        # Title Logic 
        title = "Mean Light Levels \n"
        if use_NNx_RxD == "all": 
            title = title + "All r" + str(params['dimension']).lower() + " and NN "
            
        elif use_NNx_RxD == "RxD":
            title = title + "r" + str(params['dimension']).lower() + u"\u220A"
            title = title + "["
            for mm in lvar:
                if mm != lvar[-1]:
                    title = title + str(mm) + ","
                else: 
                    title = title + str(mm) 
            title = title + "] mm, "
            
        elif use_NNx_RxD == "NNx":
            title = "NN " + str(params.Nnns)
            
        if "lambda" in info['pairs']: 
            idxs = params['Nwls'] - 1 
            
            wavelengths = lambdas[idxs]
            
            title = title + str(wavelengths) + " nm"
        
        if params['useGM']:
            title = title + " ,GM"
        
        # Appearance
        fig_axes.patch.set_facecolor('black')

        
        # Axis limits 
        if np.max(x) > 150:
            fig_axes.set_ylim(np.min(y)-30,np.max(y)+30)
            fig_axes.set_xlim(np.min(x)-10,np.max(x)+10)
        else:
            fig_axes.set_ylim(np.min(y)-30,np.max(y)+10) 
            fig_axes.set_xlim(np.min(x)-40,np.max(x)+40)

        fig_axes.set_title(title, color = 'white', fontsize = 30)
        fig_axes.axes.xaxis.set_visible(False)
        fig_axes.axes.yaxis.set_visible(False)
        
        fore_color = 'white'
        plt.rcParams["xtick.color"] =  fore_color
        
        sm = plt.cm.ScalarMappable(cmap=CMAP)
        sm.set_array([])
        
        cbaxes = inset_axes(fig_axes, width="80%", height="3%", loc='lower center', bbox_to_anchor=(0.15,0.13,0.7,1), bbox_transform=fig_axes.transAxes) 
        cb = plt.colorbar(sm, cax=cbaxes, label='Values', orientation = 'horizontal')
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['0.1% max','max'])
        cb.ax.tick_params(labelsize=20)
        cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0.8)

        return data

    
    def PlotCapPhysiologyPower(data, info, fig_axes =None,params = None):
        """ 
        PlotCapPhysiologyPower A visualization of band limited power OR
        band-referenced SNR for each optode.


        "params" fields that apply to this function (and their defaults):
            fig_size    [20, 200, 1240, 420]    Default figure position vector.
            fig_handle  (none)                  Specifies a figure to target.
                                                If empty, spawns a new figure.
            dimension   '2D'                    Specifies either a 2D or 3D
                                                plot rendering.
            rlimits     (all R2D)               Limits of pair radii displayed.
            Nnns        (all NNs)               Number of NNs displayed.
            Nwls        (all WLs)               Number of WLs averaged and
                                                displayed.
            useGM       0                       Use Good Measurements.
            Cmap.P      'hot'                   Default color mapping.

        Dependencies: PLOTCAPDATA, ISTABLEVAR, APPLYCMAP.

        See Also: PLOTCAP, PLOTCAPGOODMEAS.
        
        """
        # Parameters and Initialization.
        if params == None:
            params = {}
        LineColor = 'w'
        BkgdColor = 'k'
        Nm = len(info['pairs']['Src'])
        Ns = len(np.unique(info['pairs']['Src']))
        Nd = len(np.unique(info['pairs']['Det']))
        cs = np.unique(info['pairs']['WL']) # WLs.
        llfo=np.zeros((Ns+Nd,1))

        dims = np.shape(data)
        Nt = int(dims[-1])
        NDtf = (len(dims) > 2)

        if 'init_framerate' in info['system']:
            fr=info['system']['init_framerate']
        else:
            fr=info['system']['framerate']

        # Hardcoded params for color mapping to the units scaling used here.
        # scale_order = 1e-2
        # base = 1e9
        # dr = 3
        params['mode'] = 'patch'
        params['PD'] = 1
        params['Th'] = {}
        params['Th']['P'] = 0 
        params['DR'] = 1000

        if 'Cmap' not in params or params['Cmap'] == [] or 'P' not in params['Cmap'] or params['Cmap']['P'] ==[]:
            params['Cmap'] = {}
            params['Cmap']['P'] = 'hot'
        
        if 'dimension' not in params or params['dimension'] == []:
            params['dimension'] = '2D'
        
        if 'rlimits' not in params:
            params['rlimits']=[10,30]

        if 'WL' not in params:# Choose wavelength 
            if 'lambda' in params: # Choose wavelength from actual lambda
                params['WL'] = info['pairs']['WL'][np.where(info['pairs']['lambda'] == params['lambda'])[0]]
            else:
                params['WL'] = 2
    
        if 'Nwls' not in params or params['Nwls'] == []:
            params['Nwls'] = np.transpose(cs)
        
        if 'useGM' not in params or params['useGM'] == []:
            params['useGM'] = 0
        
        if not params['useGM'] or 'MEAS' not in info or ('MEAS' in info and 'GI' not in info['MEAS']):
            GM = np.ones((Nm, 1))
        else:
            GM = info['MEAS']['GI']
        
        if 'dimension' not in params or params['dimension'] == []:
            params['dimension'] = '2D'# '2D' | '3D'
        
        if 'OD' not in params:
            params['OD'] = 0 # 0 for raw data, 1 for Optical Density'
        
        if 'type' not in params:
            params['type'] = 'SNR' # 0 for raw data, 1 for Optical Density'
        
        if 'freqs' not in params: # Freq range to find peak
            params['freqs'] = [0.5, 2.0] # Pulse band
        elif isinstance(params['freqs'], str): 
            if params['freqs'] == 'pulse':
                params['freqs'] = [0.5, 2.0] # Pulse band
            if params['freqs'] == 'fc':
                params['freqs'] = [0.009, 0.08] # Pulse band

        if 'freqsBW' not in params:
            params['freqsBW'] = [0.009, 0.08] #range from which to determine bandwidth
    
        #  N-D Input.
        if NDtf:
            data = np.reshape(data, (-1, Nt))

        #  Use only time in synchpts if present
        if 'paradigm' in info:
            if 'synchpts' in info['paradigm']:
                NsynchPts = len(info['paradigm']['synchpts'])#  set timing of data
                if NsynchPts > 2:
                    tF = info['paradigm']['synchpts'][-1] # treat these as values rather than indices (correct for indexing when used)
                    t0 = info['paradigm']['synchpts'][1]
                elif NsynchPts == 2:
                    tF = info['paradigm']['synchpts'][1]
                    t0 = info['paradigm']['synchpts'][0]
                else:
                    tF = len(data) 
                    t0 = 1
            else:
                tF = len(data)
                t0 = 1
        else:   
            tF = len(data) 
            t0 = 1

            #  Calculate power
        if not params['OD']:
            data = tx4m.logmean(data)[0]

        ftdomain, ftmag, _, _  = tx4m.fft_tts(data[:, t0-1:tF],fr)

        greaterthan = np.where(info['pairs']['r2d'] >= params['rlimits'][0],1,0)
        lessthan = np.where(info['pairs']['r2d'] <= params['rlimits'][1],1,0)
        wlengthidx = np.where(info['pairs']['WL'] == params['WL'],1,0)
        gmidx = GM.flatten()
        keep = np.logical_and(np.logical_and(np.logical_and(greaterthan, lessthan), wlengthidx),gmidx)

        # Since these are all indices, they don't need to add 1, and 0 and 1 are used instead of 1 and 2
        idxFCm=np.argmin(abs(ftdomain-params['freqsBW'][0]))  # Define freq indices    
        idxFCM=np.argmin(abs(ftdomain-params['freqsBW'][1]))    
        BWfc=round((idxFCM-idxFCm)/2)                  # Define Bandwidth 
        idxPm=np.argmin(abs(ftdomain-params['freqs'][0]))   
        idxPM=np.argmin(abs(ftdomain-params['freqs'][1]))    
        
        idxP1=np.argmax(ftmag[keep,idxPm:idxPM+1],1) # keep idxP1 off by one until floIdx
        idxP1=round(np.mean(idxP1))  # find mean peak freq from all mean; off by one at this point

        floIdx=idxPm+idxP1-BWfc+1 
        fhiIdx=idxPm+idxP1+BWfc+1 # both floIdx and fhiIdx are off by one for indexing
        if floIdx < 1:
            floIdx = 1
        Pmax=np.sum(ftmag[:,floIdx:fhiIdx+1]**2,1) # sum pulse power

        fNoise=np.setdiff1d(np.arange(idxPm-BWfc+1,idxPM+BWfc+2),np.arange(floIdx+1,fhiIdx+2)) #correct indices here to avoid concatenating off by one error
        fNoise[np.where(fNoise < 1)] = []  
        fNoise[np.where(fNoise>len(ftdomain))] = []     
        fNoiseidx = [x - 1 for x in fNoise]
        Control=np.median(ftmag[:,fNoiseidx]**2,1)*BWfc*2 
    
        if params['type'] == 'SNR':
            Plevels = 10*np.log10(Pmax/Control) # SNR in dB
        if params['type'] == 'mag':
            Plevels = Pmax/np.amax(Pmax.flatten('F'))

            # %% Populate metric for visualizations
        llfo = np.zeros((Ns+Nd))
        for s in range(1,Ns+1):
            Sgood=np.logical_and(keep, np.where(info['pairs']['Src']==s,1,0))
            if np.sum(Sgood, axis = 0) > 0:
                Cvalue = np.mean(Plevels[Sgood]) # Average across measurements
            else:
                Cvalue = 1
            llfo[s-1] = Cvalue
        for d in range(1,Nd+1):   
            Dgood=np.logical_and(keep, np.where(info['pairs']['Det']==d,1,0))
            if np.sum(Dgood,axis = 0) > 0:
                Cvalue = np.mean(Plevels[Dgood]) # Average across measurements
            else:
                Cvalue = 1
            llfo[Ns+d-1] = Cvalue
            #  Scaling and colormapping
        M=np.amax(llfo)
        params['Scale'] = M 
        # llfo=llfo-M/2
        params['Cmap']['P'] = 'hot'
        m = max([min(llfo),0]) # in the case param{} is useful, uncomment above and delete this line

        SDRGB, CMAP,_ = viz.applycmap(llfo, [], params)
        SrcRGB = SDRGB[0:Ns, :]
        DetRGB = SDRGB[Ns:, :] 
        x,y,threeDGraph, fig_axes = viz.PlotCapData(SrcRGB, DetRGB, info, fig_axes,params)

        # Add Title and colorbar.
        title = ''

        # Appearance
        fig_axes.patch.set_facecolor('black')
        
        # Axis limits 
        if np.max(x) > 150:
            fig_axes.set_ylim(np.min(y)-30,np.max(y)+30)
            fig_axes.set_xlim(np.min(x)-10,np.max(x)+10)
        else:
            fig_axes.set_ylim(np.min(y)-30,np.max(y)+10) 
            fig_axes.set_xlim(np.min(x)-40,np.max(x)+40)
        
        fig_axes.xaxis.set_visible(False)
        fig_axes.yaxis.set_visible(False)
        
        fore_color = 'white'
        plt.rcParams["xtick.color"] =  fore_color
        
        sm = plt.cm.ScalarMappable(cmap=CMAP)
        sm.set_array([])
        
        cbaxes = inset_axes(fig_axes, width="80%", height="3%", loc='lower center', bbox_to_anchor=(0.15,0.13,0.7,1), bbox_transform=fig_axes.transAxes) 
        cb = plt.colorbar(sm, cax=cbaxes, label='Values', orientation = 'horizontal')
        cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0.8)
        cb.ax.tick_params(labelsize=20)

        
        if params['type'] == 'SNR':
            title = title + 'Mean Band-limited SNR\n' + 'r'+ str(np.char.lower(params['dimension'])) + ' \u220A [' + str(params['rlimits'][0]) + ', ' + str(params['rlimits'][1]) + '] mm; f\u220A ['
            title = title + str(params['freqs'][0]) + ', ' + str(params['freqs'][1]) + '] Hz'  

            cb.set_ticks([0, 0.5, 1])
            cb.set_ticklabels([str(round(m,4)),r'$SNR_{dB}$',str(round(M,4))])

        if params['type'] == 'mag':
            title = title + 'Mean FFT Power\n' + 'r'+str(np.char.lower(params['dimension']))+ ' \u220A [' + str(params['rlimits'][0]) + ', ' + str(params['rlimits'][1]) + '] mm; f\u220A ['
            title = title + str(params['freqs'][0]) + ', ' + str(params['freqs'][1]) + '] Hz'
    
            cb.set_ticks([0, 0.5, 1])
            cb.set_ticklabels([str(round(m,4)),'Max Relative Power',str(round(M,4))])

        fig_axes.set_title(title, color = 'white', fontsize = 30, pad = 0.1)
        fig_axes.set_facecolor('black')
        
        return Plevels

    
    def PlotTimeTraceData(data, time, params = None, fig_axes = None, coordinates = [1,1]): # specify which plot in subplot the figure is going to with ax - add doc later
        """ 
        PLOTTIMETRACEDATA A basic time traces plotting function.

        PLOTTIMETRACEDATA(data, time) takes a light-level array "data" of the
        MEAS x TIME format, and plots its time traces.

        PLOTTIMETRACEDATA(data, time, params) allows the user to specify
        parameters for plot creation.
        
        h = PLOTTIMETRACEDATA(...) passes the handles of the plot line objects
        created.

        "params" fields that apply to this function (and their defaults):
            fig_size    [200, 200, 560, 420]    Default figure position vector.
            fig_handle  (none)                  Specifies a figure to target.
                                                If empty, spawns a new figure.
            xlimits     'auto'                  Limits of x-axis.
            xscale      'linear'                Scaling of x-axis.
            ylimits     'auto'                  Limits of y-axis.
            yscale      'linear'                Scaling of y-axis.

        See Also: PLOTTIMETRACEALLMEAS, PLOTTIMETRACEMEAN.
        
        """
        ## Parameters and Initialization.
        if params == None:
            params = {}

        LineColor = 'w'
        BkgdColor = 'k'

        dims = data.shape
        Nt = dims[-1]
        NDtf = (len(dims) > 2)
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches

        if params is None:
            params = {}
        if 'fig_size' not in params or params['fig_size'] == []:
            params['fig_size'] = [560*px, 420*px]

        if 'xlimits' not in params or params['xlimits'] == []:
            params['xlimits'] = []
        
        if 'xscale' not in params or params['xscale'] is None:
            params['xscale'] = 'linear'
        
        if 'ylimits' not in params or params['ylimits'] == [] or (all(all(data > params['ylimits'][1]))) and all(all(data<params['ylimits'][0])):
            params['ylimits'] = []

        if 'yscale' not in params or params['yscale'] is None:
            params['yscale'] = 'linear'

        if time is None:
            time = np.arange(1,Nt,1)

        if NDtf:
            data = np.reshape(data, len(data)/Nt, Nt)

    #   Plot Data.
        if data is None: 
            data = np.zeros(time.shape) 
            
        for plot in data: 
            fig_axes[coordinates[0],coordinates[1]].plot(time, plot, linewidth=0.15)
        for ax in fig_axes.flat:
        # Setting the background color of the plot 
            ax.set_facecolor("black")
            ax.set_yscale(params['yscale'])
            ax.set_xscale(params['xscale'])
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.margins(x = 0)
            ax.margins(y = 0)
        return ax

    
    def vol2surf_mesh(Smesh, volume, dim, params = None):
        """
        VOL2SURF_MESH Interpolates volumetric data onto a surface mesh.

        Smesh = VOL2SURF_MESH(mesh_in, volume, dim) takes the mesh "Smesh"
        and interpolates the values of the volumetric data "volume" at the
        mesh's surface, using the spatial information in "dim". These values
        are output as "Smesh".

        Smesh = VOL2SURF_MESH(Smesh, volume, dim, params) allows the user
        to specify parameters for plot creation.

        "params" fields that apply to this function (and their defaults):
            OL      0   If "overlap" data is presented (OL==1), this sets the
                        interpolation method to "nearest". Default is "linear".

        See Also: PLOTINTERPSURFMESH, GOOD_VOX2VOL, AFFINE3D_IMG.
        """

        ## Parameters and Initialization.
        if params == None:
            params = {}

        size_vol = volume.shape
        if len(size_vol) <= 3:
            Ncols = 1
        else:
            Ncols = size_vol[3]
        Ncoords = Smesh['nodes'].shape[0]
        Smesh['data'] = np.zeros((Ncoords, Ncols))
        extrapval = 0

        nVx = dim['nVx']
        nVy = dim['nVy']
        nVz = dim['nVz']
        dr = dim['mmppix']
        center = dim['center']

        if not 'OL' in params or params['OL'] == False:
            params['OL'] = 0
        if 'OL' in params and params['OL'] == 1:
            method = 'nearest'
        else:
            method = 'linear'

        buffer = 2

        ## Define coordinate space of volumetric data
        X = np.multiply(dr[0], np.arange(nVx, 0, -1)) - center[0] # changed stop index to be 0 instead of 1 to get the correct number of elements in vector bc of python 0 indexing
        Y = np.multiply(dr[1], np.arange(nVy, 0, -1)) - center[1]
        Z = np.multiply(dr[2], np.arange(nVz, 0, -1)) - center[2]

        ## Get coordinates for surface mesh
        x = Smesh['nodes'][:,0]
        y = Smesh['nodes'][:,1]
        z = Smesh['nodes'][:,2]


        ## Correct for nodes just outside of volume (MNI and TT atlas space cuts off occipital pole and part of dorsal tip and lateral extremes).
        x[np.logical_and((x < min(X)), (x > min(X) - buffer))] =  min(X)
        x[np.logical_and((x > max(X)), (x > max(X) - buffer))] =  max(X)

        y[np.logical_and((y < min(Y)), (y > min(Y) - buffer))] =  min(Y)
        y[np.logical_and((y > max(Y)), (y > max(Y) - buffer))] =  max(Y)

        z[np.logical_and((z < min(Z)), (z > min(Z) - buffer))] =  min(Z)
        z[np.logical_and((z > max(Z)), (z > max(Z) - buffer))] =  max(Z)

        ## Interpolate
        X = np.flip(X)
        volume = np.flip(volume, axis = 0)
        for k in range(0, Ncols):
            Smesh['data'] = scipy.interpolate.interpn((X, Y, Z), np.squeeze(volume[:,:,:]), np.array([x, y, z]).T, method = method, bounds_error = False, fill_value = extrapval)
        
        return Smesh