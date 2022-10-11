# -*- coding: utf-8 -*-
r"""
================================
NeuroDOTPy Preprocessing Pipeline
=================================


This script combines the Preprocessing and Reconstruction pipelines. A set of sample Data files exist in the /Data directory of the toolbox.

For each data set, there is an example *.pptx with selected visualizations. The NeuroDOT_Tutorial_Full_Data_Processing.pptx uses NeuroDOT_Data_Sample_CCW1.mat.


"""  

# sphinx_gallery_thumbnail_number = 1

# %%
# Importage
# --------------------------------------------------

import sys,os
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy.matlib as nm

from matplotlib.pyplot import colorbar, colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Add NeuroDOT library to the path
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'neuro_dot'))

from Visualizations import viz
from Spatial_Transforms import sx4m
from Temporal_Transforms import tx4m
from Light_Modeling import lmdl
from File_IO import io
from Analysis import anlys
from Matlab_Equivalent_Functions import matlab
from Reconstruction import recon

# %%
# Grab the data
# --------------------------------------------------

participant_data = 'NeuroDOT_Data_Sample_CCW1.mat' # Name of your data file, or one of the NeuroDOT data samples in the 'Data' folder
data_path = os.path.join(os.path.dirname(sys.path[0]),'Data',participant_data)
data = io.loadmat(data_path)['data']
__info = io.loadmat(data_path)['info']             # adding __info makes info private, so any changes will be function-specific
flags = io.loadmat(data_path)['flags']
E = None
MNI = None

# %%
# Set Parameters for Preprocessing
# ---------------------------------------------------

# Default parameters
params = {'bthresh':0.075,'det':1, 'highpass':1, 'lowpass1':1, 'ssr':1, 'lowpass2':1, 'DoGVTD':1, 'resample': 1, 'omega_hp': 0.02, 'omega_lp1': 1}
    


