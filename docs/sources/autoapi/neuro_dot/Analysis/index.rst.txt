:py:mod:`neuro_dot.Analysis`
============================

.. py:module:: neuro_dot.Analysis


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Analysis.BlockAverage
   neuro_dot.Analysis.CalcGVTD
   neuro_dot.Analysis.FindGoodMeas
   neuro_dot.Analysis.normcND
   neuro_dot.Analysis.normrND



.. py:function:: BlockAverage(data_in, pulse, dt, Tkeep=0)

   BLOCKAVERAGE Averages data by stimulus blocks.

   data_out = BLOCKAVERAGE(data_in, pulse, dt) takes a data array "data_in" 
   and uses the pulse and dt information to cut that data timewise into 
   blocks of equal length (dt), which are then averaged together and 
   output as "data_out".

   Tkeep is a temporal mask. Any time points with a zero in this vector is
   set to NaN.


.. py:function:: CalcGVTD(data)

   CalcGVTD calculates the Root Mean Square across measurements (log-mean light levels or voxels) of the temporal derivative. 

   The data is assumed to have measurements in the first and time in the last 
   dimension. 

   Any selection of measurement type or voxel index must be done
   outside of this function.


.. py:function:: FindGoodMeas(data, info_in, bthresh=0.075)

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


.. py:function:: normcND(data)

   NORMCND returns a column-normed matrix. It is assumed that the matrix is 2D.


.. py:function:: normrND(data)

   NORMRND returns a row-normed matrix. It is assumed that the matrix is 2D. Updated for broader compatability.


