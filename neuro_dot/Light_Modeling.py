# General imports
import sys
import math

import scipy.io as spio
import scipy.signal as sig
import numpy as np
import numpy.linalg as lna
import numpy.linalg as la
import deepdiff as dd
import pprint as pp
import normalize_easy as norm
import fractions as fr
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patch
import matplotlib.colors as colors
import matplotlib.pylab as plb
import scipy.optimize as ops
import scipy.ndimage as ndi
import scipy.interpolate
import numpy.matlib as mlb
import numpy.matlib as nm
import functools as ft
import statistics as stats

from math import trunc
from pickle import NONE
from numpy import float64, matrix
from numpy.lib.shape_base import expand_dims

class lmdl:

    def makeFlatFieldRecon(A, iA):
        ## Flat Field Perturbation
        ff = np.ones(shape = (np.shape(A)[1], 1))

        ## Simulated Measurements
        ysim = A.astype(np.float64) @ ff.astype(np.float64)

        ## Flat Field Reconstruction
        Asens = iA.astype(np.float64) @ ysim.astype(np.float64)

        return Asens