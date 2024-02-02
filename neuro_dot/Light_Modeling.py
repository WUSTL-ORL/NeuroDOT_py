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
import fractions as fr
import sympy as sym
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
from numpy.linalg import norm

from Light_Modeling import lmdl

class lmdl:

    def calc_NN(info_in, dr):
        ## Parameters and Initialization
        info_out = info_in
        if dr is None:
            dr = 10 #default = 10mm minimum separation for SD to be grouped into different neighbors
        Nm = len(info_in['pairs']['r3d'])
        NN=np.zeros((Nm)) #initialize NN vector
        ## Calculate NN's
        RadMaxR=np.ceil(np.max(info_in['pairs']['r3d'])) #maximum SD separation across all SD pairs
        d=0; # s-d distance
        c=0; # which nn are we on?
        while np.any(info_in['pairs']['r3d']>d): # as long as there are still s-d pairs left to group
            if ((d==0) & (dr>9)):
                nnkeep=np.argwhere(np.logical_and(np.where(info_in['pairs']['r3d']>=d,1,0), np.where(info_in['pairs']['r3d']<(d+(2*dr)),1,0)))
                d=d+dr
                nnkeep = nnkeep[:,0]
            else:            
                nnkeep=np.argwhere(np.logical_and(np.where(info_in['pairs']['r3d']>=d,1,0), np.where(info_in['pairs']['r3d']<(d+dr),1,0)))
                nnkeep = nnkeep[:,0]
            # find pairs within 1 mm range
            if nnkeep != []: # if we find pairs
                c=c+1 # increment nn count
                NN[nnkeep]=c
                if c>RadMaxR: 
                    break # stop at nn9
            d=d+dr # incremement search distance
        info_out['pairs']['NN']=NN
        return info_out

    def Generate_pad_from_grid(grid, params, info):
        ## Parameters and Initialization

        # Params and output structure
        if info is None:
            info = dict()
        if params is None:
            params = dict()
        if not('dr' in params):
            params['dr']=10 # defaults to 10mm
        else:
            dr = params['dr']
        if params['lambda'] is None:
            params['lambda'] = [750,850]
        if not('Mod' in params):
            params['Mod'] = 'CW'
        if not('pos2' in params):
            params['pos2'] = 0

        # Optode positions
        # Make generalizable (can be either 2D or 3D) optode pos
        # Defaults to 3D pos if 2D pos not part of input grid structure
        if not('spos' in grid):
            if 'spos2' in grid and 'spos2'!= None:
                grid['spos'] = grid['spos2']
            else:
                grid['spos'] = grid['spos3']
        

        if 'dpos' in grid:
            if 'dpos2' in grid and 'dpos2'!= None:
                grid['dpos'] = grid['dpos2']
            else:
                grid['dpos'] = grid['dpos3']
        # If 3D not supplied as input, set to 2D pos where col3 is all zeros
        if not('spos3' in grid):
            zarray = np.zeros(len(grid['spos']),1)
            grid['spos3'] = np.hstack(grid['spos'],zarray)
        
        if not('dpos3' in grid):
            zarray = np.zeros(len(grid['spos']),1)
            grid['dpos3'] = np.hstack(grid['dpos'],zarray)

        # Calculate number of sources, detectors, measurements, and wavelengths
        Ns = np.shape(grid['spos3'])[0]
        Nd = np.shape(grid['dpos3'])[0]
        Nm = Ns*Nd
        Nwl = len(params['lambda'])

        # Initialize SD separations and measurement list
        # Note: these are all for a single wavelength, and will get replicated 
        #   in a below section of the function for all other WL
        r2 = np.zeros((Nm,1)) #2D SD separations
        r3 = np.zeros((Nm,1)) #3D SD separations
        measList = np.zeros((Nm,2)) #basic measurement list that only has [Src, Det]


        ## Populate info.optodes structure

        # Detectors
        if 'CapName' in params:
            info['optodes']['CapName'] = params['CapName']
        if 'dpos3' in grid:
            info['optodes']['dpos3'] = grid['dpos3']
        if 'dpos2' in grid and 'dpos2' != None: # if 2d detector positions and dpos2 isn't none
            info['optodes']['dpos2'] = grid['dpos2']
        elif 'dpos' in grid: # if no 2d detector positions and dpos is there
            info['optodes']['dpos2'] = grid['dpos']
        else:           
            info['optodes']['dpos2'] = grid['dpos3'] # if no 2d detector pos and no dpos
        

        # Sources
        if 'spos3' in grid:
            info['optodes']['spos3'] = grid['spos3']
        if 'spos2' in grid and 'spos2'!= None:
            info['optodes']['spos2'] = grid['spos2']
        elif 'spos' in grid:
            info['optodes']['spos2'] = grid['spos']
        else:
            info['optodes']['spos2'] = grid['spos3']

        ## Make Measlist, r3d, and r2d
        m = 0
    
        for d in range(1, Nd+1):# 1:Nd:
            for s in range(1, Ns+1):#= 1:Ns
                measList[m,0] = s
                measList[m,1] = d
                r2[m] = norm(info['optodes']['spos2'][s-1,:]-info['optodes']['dpos2'][d-1,:])
                # print('R2: ', r2)
                r3[m] = norm(info['optodes']['spos3'][s-1,:]-info['optodes']['dpos3'][d-1,:])
                m+=1

        ## Populate info.pairs structure
        info = dict()
        info['pairs'] = dict()
        info['pairs']['Src'] = np.tile(measList[:,0],[Nwl,1]).flatten()
        print('src in generate pad from grid: ', info['pairs']['Src'])
        info['pairs']['Det'] = np.tile(measList[:,1],[Nwl,1]).flatten()
        # info.pairs.NN will be created and populated below
        narray = np.zeros(np.size(info['pairs']['Src']))
        narray2 = np.zeros(np.size(info['pairs']['Src']))
        narray[:] = np.nan
        narray2[:] = np.nan
        info['pairs']['WL'] = narray2
        info['pairs']['lambda'] = narray
        info['pairs']['Mod'] = np.tile(params['Mod'],[Nm*Nwl,1])
        info['pairs']['r2d'] = np.tile(r2,[Nwl,1])
        info['pairs']['r3d'] = np.tile(r3,[Nwl,1])

        # Make sure WL and Lambda reflect both wavelengths
        for j in range(1, Nwl+1): 
            info['pairs']['WL'][Nm*(j-1):Nm*j] = np.multiply(np.ones((Nm)),j)
            info['pairs']['lambda'][Nm*(j-1)+1:Nm*j] = params['lambda'][j-1]


        ## Populate NN's 
        info = lmdl.calc_NN(info,params['dr']) #updated number of measurment calculation within calc_NN to be based on length of r3d

        return info
        
    def makeFlatFieldRecon(A, iA):
        ## Flat Field Perturbation
        ff = np.ones(shape = (np.shape(A)[1], 1))

        ## Simulated Measurements
        ysim = A.astype(np.float64) @ ff.astype(np.float64)

        ## Flat Field Reconstruction
        Asens = iA.astype(np.float64) @ ysim.astype(np.float64)

        return Asens