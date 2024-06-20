# Import all necessary Python modules
import numpy as np 
import scipy.io as sio
import sys,os  
import matplotlib.pyplot as plt  
import math
import numpy.matlib as nm
import pylab
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from matplotlib.cm import jet
import copy
# Add all relevant paths (e.g. folders containing NeuroDOT functions and data)



# Import all functions that will be used in the pipeline (consider using __init__ file instead)
# In __init__ file in 'Individual Functions' directory, change the path to your system folders
# containing each function folder


import neuro_dot as ndot

def DynamicFilter(input_data, info_in, params, mode, save = 'no', pathToSave = './'):
    '''
    This function is used to perform each step of the NeuroDOT PreProcessing Pipeline.

    The input "mode" is used to decide which step of the pipeline to perform, as well as 
    which figures to generate.

    Modes:
        :fft_lml: display Fourier transform of the logmean light levels
        :lml: display time traces of logmean light levels
        :fft_dt: display Fourier transform of detrended data
        :fft_dt_hp: display Fourier transform of detrended and high-pass filtered data
        :fft_superficial_signal: display Fourier transform of data which has undergone superficial signal regression
        :high_pass: display high-pass filtered data time traces
        :low_pass: display low-pass filtered data time traces
        :low_pass_fft: display Fourier transform of low-pass filtered data  
        :fft_lowpass_2: display Fourier transform of twice-low-pass filtered data  
        :superficial_signal: display time traces of data which has undergone superficial signal regression
        :fft_resample: display Fourier transform of resampled data
        :resample: display time traces of resampled data
        :ba: display time traces of block-averaged data
        :fft_ba: display Fourier transform of block-averaged data
    '''
    lmdata = ndot.logmean(input_data)[0]
    __info = copy.deepcopy(info_in)
    keep = np.logical_and(np.logical_and(np.where(__info['pairs']['WL'] == 2,1,0), np.where(__info['pairs']['r2d'] < 40,1,0)), __info['MEAS']['GI']) # measurements to include
    keepd1=np.logical_and(np.logical_and(__info['MEAS']['GI'], np.where(__info['pairs']['r2d']<20,1,0)), np.where(__info['pairs']['WL']==2,1,0))
    keepd2=np.logical_and(np.logical_and(np.logical_and(__info['MEAS']['GI'], np.where(__info['pairs']['r2d']>=20,1,0)), np.where(__info['pairs']['r2d']<30,1,0)), np.where(__info['pairs']['WL']==2,1,0))
    keepd3=np.logical_and(np.logical_and(np.logical_and(__info['MEAS']['GI'], np.where(__info['pairs']['r2d']>=30,1,0)), np.where(__info['pairs']['r2d']<40,1,0)), np.where(__info['pairs']['WL']==2,1,0))

    
    if mode == 'fft_lml':
        figdata = lmdata
        keep = np.logical_and(np.logical_and(np.where(__info['pairs']['WL'] == 2,1,0), np.where(__info['pairs']['r2d'] < 40,1,0)), __info['MEAS']['GI']) # measurements to include

    elif mode == 'lml':
        figdata = lmdata
       
    elif mode == 'fft_dt':
        figdata = ndot.detrend_tts(lmdata)

    elif mode == 'fft_dt_hp':
        figdata = ndot.detrend_tts(lmdata)
        figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])

    elif mode == 'fft_superficial_signal':
        ## Superficial Signal Regression
        if params['det']==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:
            figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])
        if params['lowpass1'] == 1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'], __info['system']['framerate']) 
        hem = ndot.gethem(figdata, __info)
        figdata, _ = ndot.regcorr(figdata, __info, hem)

    elif mode == 'high_pass':
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])

    elif mode == 'low_pass':
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:
            figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])
        figdata = ndot.lowpass(figdata, params['omega_lp1'], __info['system']['framerate']) 
        
    elif mode == 'low_pass_fft':
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:
            figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])
        figdata = ndot.lowpass(figdata, params['omega_lp1'], __info['system']['framerate'])

    elif mode == 'fft_low_pass2':
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:
            figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'], __info['system']['framerate']) 
        if params['ssr'] ==1:
            hem = ndot.gethem(figdata, __info)
            figdata, _ = ndot.regcorr(figdata, __info, hem)
        figdata = ndot.lowpass(figdata, params['omega_lp2'], __info['system']['framerate'])

    elif mode == 'superficial_signal':
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:    
            figdata = ndot.highpass(figdata, params['omega_hp'], __info['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'], __info['system']['framerate']) 
        hem = ndot.gethem(figdata, __info)
        figdata, _ = ndot.regcorr(figdata, __info, hem)
        # % SSRdata = lp1data; % example to ignore SSR

    elif mode == 'fft_resample':
        __info_new = info_in.copy()
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:   
            figdata = ndot.highpass(figdata, params['omega_hp'], __info_new['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'], __info_new['system']['framerate']) 
        if params['ssr'] ==1:
            hem = ndot.gethem(figdata, __info_new)
            figdata, __info_out = ndot.regcorr(figdata, __info_new, hem)
        if params['lowpass2'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp2'], __info_new['system']['framerate'])
        figdata, __info = ndot.resample_tts(figdata, __info_new, params['omega_resample'], params['rstol'])

    elif mode == 'resample':
        __info_new = copy.deepcopy(info_in)
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1:  
            figdata = ndot.highpass(figdata, params['omega_hp'], __info_new['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'], __info_new['system']['framerate']) 
        if params['ssr'] ==1:
            hem = ndot.gethem(figdata, __info_new)
            figdata, __info_out = ndot.regcorr(figdata, __info_new, hem)
        if params['lowpass2'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp2'], __info_new['system']['framerate'])
        figdata, __info = ndot.resample_tts(figdata, __info_new, params['omega_resample'], params['rstol'])

    elif mode == 'ba':
        __info_new = __info.copy()
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1: 
            figdata = ndot.highpass(figdata,params['omega_hp'],  __info_new['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'],  __info_new['system']['framerate']) 
        if params['ssr'] ==1:
            hem = ndot.gethem(figdata, __info_new)
            figdata, __info_out = ndot.regcorr(figdata, __info_new, hem)
        if params['lowpass2'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp2'],  __info_new['system']['framerate'])
        if params['resample'] ==1:
            figdata, __info = ndot.resample_tts(figdata, __info_new, params['omega_resample'], params['rstol'])
        synchs = __info['paradigm']['Pulse_2']-1
        figdata,BSTD_out, BT_out, blocks = ndot.BlockAverage(figdata, __info['paradigm']['synchpts'][synchs], params['dt'])
    elif mode == 'fft_ba':
        __info_new = __info.copy()
        if params['det'] ==1:
            figdata = ndot.detrend_tts(lmdata)
        if params['highpass']==1: 
            figdata = ndot.highpass(figdata, params['omega_hp'],  __info_new['system']['framerate'])
        if params['lowpass1'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp1'],  __info_new['system']['framerate']) 
        if params['ssr'] ==1:
            hem = ndot.gethem(figdata, __info_new)
            figdata, __info_out = ndot.regcorr(figdata, __info_new, hem)
        if params['lowpass2'] ==1:
            figdata = ndot.lowpass(figdata, params['omega_lp2'],  __info_new['system']['framerate'])
        if params['resample'] ==1:
            figdata, __info = ndot.resample_tts(figdata, __info_new, params['omega_resample'], params['rstol'])
        synchs = __info['paradigm']['Pulse_2']-1
        figdata, BSTD_out, BT_out, blocks = ndot.BlockAverage(figdata, __info['paradigm']['synchpts'][synchs], params['dt'])
        
   

    if 'fft' in mode:
        if 'ba' not in mode:
            __info['GVTD'] = ndot.CalcGVTD((figdata[np.logical_and(__info['MEAS']['GI'],np.where(__info['pairs']['r2d'] < 20,1,0)[0])]))
            ndot.nlrGrayPlots_220324(figdata,__info)
        else:   
            ndot.nlrGrayPlots_220324(figdata,__info, mode = 'ba')

        fig1 = plt.figure(dpi = 150)
        fig1.set_size_inches(6, 9)
        gs1 = gridspec.GridSpec(3,1)
        ax1 =  plt.subplot(gs1[0,0])
        ax2 =  plt.subplot(gs1[1,0])
        ax3 =  plt.subplot(gs1[2,0])

        xplot = np.transpose(np.reshape(np.mean(np.transpose(figdata[keep,:]),1), (len(np.mean(np.transpose(figdata[keep,:]),1)),1)))
        ftdomain, ftmag,_,_ = ndot.fft_tts(xplot,__info['system']['framerate']) # Generate average spectrum
        ftmag = np.reshape(ftmag, (len(ftdomain)))  

        ax1.plot(np.transpose(figdata[keep,:]),linewidth = 0.2) # plot signals 
        ax1.set_xlabel('Time (samples)',labelpad = 5)
        ax1.set_ylabel('log(\u03A6/$\u03A6_{0}$)') 
        ax1.xaxis.set_tick_params(color='black')
        ax1.tick_params(axis='x', colors='black', pad = 10, size = 5)
        ax1.set_ylim([1.25*np.amin(figdata[keep,:]),1.25*np.amax(abs(figdata[keep,:]))])
         
        ax1.set_xlim([0, len(figdata[keepd1][1])])

        im2 = ax2.imshow(figdata[keep,:], aspect = 'auto')
        ax2.set_xlabel('Time (samples)',labelpad = 5)
        ax2.set_ylabel('Measurement #') # show signals as image
        ax2.xaxis.set_tick_params(color='black')
        ax2.tick_params(axis='x', colors='black', pad = 10, size = 5)

        axins1 = inset_axes(ax2,
                        width="100%",  # width = 50% of parent_bbox width
                        height="20%",  # height : 5%
                        loc='upper center',
                        bbox_to_anchor=(0, 0.5, 1, 1),
                        bbox_transform=ax2.transAxes)
        
        cb = fig1.colorbar(im2, cax=axins1, orientation="horizontal", drawedges=False)
        cb.ax.xaxis.set_tick_params(color="black", pad = 0, length = 2)
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color="black", fontsize = 8)
        cb.outline.set_edgecolor('black')
        cb.outline.set_linewidth(0.5)
            
        ax3.semilogx(ftdomain,ftmag, linewidth = 0.5) # plot vs. log frequency
        ax3.set_xlabel('Frequency (Hz)', labelpad = 5)
        ax3.set_ylabel('|X(f)|') # plot vs. log frequency
        ax3.set_xlim([1e-3, 10])
        ax3.xaxis.set_tick_params(color='black')
        ax3.tick_params(axis='x', colors='black', pad = 10, size = 5)

        plt.subplots_adjust(hspace = 1)

        

    else:

         # Generate 3 subplots
        fig1 = plt.figure(dpi = 150)
        fig1.set_size_inches(6, 6)
        gs1 = gridspec.GridSpec(3,1)
        ax1 =  plt.subplot(gs1[0,0])
        ax2 =  plt.subplot(gs1[1,0])
        ax3 =  plt.subplot(gs1[2,0])
        # plt.subplots_adjust(hspace=1.2)

        ax1.plot(np.transpose(figdata[keepd1,:]), linewidth = 0.2)
        ax1.set_ylabel('$R_{sd}$ < 20 mm', fontsize = 5)
        ax1.set_ylim([np.amin(figdata[keepd1])*1.25, np.amax(figdata[keepd1])*1.25])
        ax1.set_xlim([0, len(figdata[keepd1][1])-1])
        ax1.xaxis.set_tick_params(color='black')
        ax1.tick_params(axis='x', colors='black')

        ax2.plot(np.transpose(figdata[keepd2,:]), linewidth = 0.2)
        ax2.set_ylabel('$R_{sd}$ \u220A [20 30] mm', fontsize = 5)
        ax2.set_xlim([0, len(figdata[keepd2][1])-1])
        ax2.xaxis.set_tick_params(color='black')
        ax2.tick_params(axis='x', colors='black')

        # ax2.set_xlim([0, 2600])

        ax3.plot(np.transpose(figdata[keepd3,:]), linewidth = 0.2)
        ax3.set_ylabel('$R_{sd}$ \u220A [30 40] mm',fontsize = 5)
        ax3.set_xlabel('Time (samples)',labelpad = 12)
        ax3.set_xlim([0, len(figdata[keepd3][1])-1])
        ax3.xaxis.set_tick_params(color='black')
        ax3.tick_params(axis='x', colors='black')

        # ax3.set_xlim([0, 2600])

    if mode == 'resample':
        ax1.set_xlim([160, 160+36*3])
        ax2.set_xlim([160, 160+36*3])
        ax3.set_xlim([160, 160+36*3])
    # maxx = np.max(figdata[keepd1,:])
    # if maxx < 0.2:
    #     ax1.set_xlim([np.min(figdata[keepd1,:]), np.max(figdata[keepd1,:])])
    #     ax2.set_xlim([np.min(figdata[keepd2,:]), np.max(figdata[keepd2,:])])
    #     ax3.set_xlim([np.min(figdata[keepd3,:]), np.max(figdata[keepd3,:])])

    tag = __info['io']['tag']
    filename = pathToSave + mode +'_' + tag + '.png'  
    print(filename)  
    if save == 'yes':
        plt.savefig(filename,format = 'png',facecolor='white')
    if 'ba' in mode or 'resample' in mode:
        return __info_new
