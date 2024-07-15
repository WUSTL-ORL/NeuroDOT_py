:py:mod:`neuro_dot.DynamicFilter`
=================================

.. py:module:: neuro_dot.DynamicFilter


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.DynamicFilter.DynamicFilter



.. py:function:: DynamicFilter(input_data, info_in, params, mode, save='no', pathToSave='./')

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


