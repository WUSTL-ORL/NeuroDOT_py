:py:mod:`neuro_dot.Light_Modeling`
==================================

.. py:module:: neuro_dot.Light_Modeling


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Light_Modeling.calc_NN
   neuro_dot.Light_Modeling.Generate_pad_from_grid
   neuro_dot.Light_Modeling.makeFlatFieldRecon



.. py:function:: calc_NN(info_in, dr)

   CALC_NN Calculates the Nearest Neigbor value for all measurement pairs.

   Inputs:
       :info_in: info structure containing data measurement list
       :dr: minimum separation for sources and detectors to be grouped into. (default = 10 mm) NOTE: distances are in millimeters
       
   Outputs:
       :info_out: info structure containing updated data measurement list with "info.pairs.NN"


.. py:function:: Generate_pad_from_grid(grid, params, info)

   GENERATE_PAD_FROM_GRID Generates info structures "optodes" and "pairs" from a given "grid". 


   The input: "grid" must have fields that list the spatial locations of 
   sources and detectors in 3D: spos3, dpos3.

   The input: "params" can be used to pass in mod type (default is 'CW'
   but can be the modulation frequency if fd) and wavelength(s) of the 
   data in field 'lambda'.

   Params:
       :dr: Minimum separation for sources and detectors to be grouped into different neighbors.
       :lambda: Wavelengths of the light. Any number of comma-separated values is allowed. Default: [750,850].
       :Mod: Modulation type or frequency. 
           Can be 'CW or 'FD' or can be the actual modulation frequency (e.g., 0 or 200) in MHz.
       :pos2: Flag which determines NN classification. 
           Defaults to 0, where 3D coordinates are used. 
           If set to 1, 2D coordinates will be used for NN classification.
       :CapName: Name for your pad file.


.. py:function:: makeFlatFieldRecon(A, iA)

   MAKEFLATFIELDRECON Generates the flat field reconstruction of a given "A" sensitivity matrix.

   Inputs:
       :A: "A" sensitivity matrix
       :iA: inverted "A" sensitivity matrix

   Outputs:
       :Asens: flat field reconstruction of a given "A" sensitivity matrix    


