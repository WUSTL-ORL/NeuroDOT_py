:py:mod:`neuro_dot.Matlab_Equivalent_Functions`
===============================================

.. py:module:: neuro_dot.Matlab_Equivalent_Functions


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Matlab_Equivalent_Functions.rms_py



.. py:function:: rms_py(rms_input)

   RMS_PY Computes the RMS value for each column of a row vector.

   For matrices (N x M), rms_py(rms_input) is a row vector containing the RMS value from each column

   For complex input, rms_py separates the real and imaginary portions of each column of rms_input, squares them,
   and then takes the square root of the mean of that value

   For real input, the root mean square is calculated as follows:


