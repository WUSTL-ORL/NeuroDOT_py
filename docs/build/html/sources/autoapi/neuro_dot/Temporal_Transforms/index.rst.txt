:py:mod:`neuro_dot.Temporal_Transforms`
=======================================

.. py:module:: neuro_dot.Temporal_Transforms


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Temporal_Transforms.detrend_tts
   neuro_dot.Temporal_Transforms.nextpow2
   neuro_dot.Temporal_Transforms.fft_tts
   neuro_dot.Temporal_Transforms.gethem
   neuro_dot.Temporal_Transforms.highpass
   neuro_dot.Temporal_Transforms.logmean
   neuro_dot.Temporal_Transforms.lowpass
   neuro_dot.Temporal_Transforms.regcorr
   neuro_dot.Temporal_Transforms.resample_tts



.. py:function:: detrend_tts(data_in)

   DETREND_TTS Performs linear detrending.

   data_out = DETREND_TTS(data_in) takes a raw light-level data array
   "data_in" of the format MEAS x TIME and removes the straight-line fit
   along the TIME dimension from each measurement, returning it as
   "data_out".

   See Also: LOGMEAN.


.. py:function:: nextpow2(N)

   NEXTPOW2 Finds the next power of 2, given an integer input.


.. py:function:: fft_tts(data, framerate)

   FFT_TTS Computes the Fourier Transform of a time-domain input.
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


.. py:function:: gethem(data, info, sel_type='r2d', value=[10, 16])

   GETHEM Calculates the mean across a set of measurements.

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


.. py:function:: highpass(data_in, omegaHz, frate, params=None)

   HIGHPASS Applies a zero-phase digital highpass filter.
   data_out = HIGHPASS(data_in, omegaHz, frate) takes a light-level array
   "data_in" in the MEAS x TIME format and applies to it a
   forward-backward zero-phase digital highpass filter at a Nyquist cutoff
   frequency of "omegaHz * (2 * frate)", returning it as "data_out".

   This function also removes the linear component of the input data.

   See Also: LOWPASS, LOGMEAN, FILTFILT.


.. py:function:: logmean(data_in)

   LOGMEAN Computes the log-ratio of raw intensity data.

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


.. py:function:: lowpass(data_in, omegaHz, frate, params=None)

   LOWPASS Applies a zero-phase digital lowpass filter.

   data_out = LOWPASS(data_in, omegaHz, frate) takes a light-level array
   "data_in" in the MEAS x TIME format and applies to it a
   forward-backward zero-phase digital lowpass filter at a Nyquist cutoff
   frequency of "omegaHz * (2 * frate)", returning it as "data_out".

   This function also removes the linear component of the input data.

   See Also: HIGHPASS, LOGMEAN.


.. py:function:: regcorr(data_in, info, hem)

   REGCORR Performs regression correction by wavelength.

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

       R = (<y_{in},y_{r}>/(|y_{in}|*|y_{r}|)).

   See Also: GETHEM, DETREND_TTS.


.. py:function:: resample_tts(data_in, info_in, omega_resample=1, tol=0.001, framerate=0)

   RESAMPLE_TTS Resamples data while maintaining linear signal components.

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


