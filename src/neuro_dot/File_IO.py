# General imports
import sys
import math
import os
import mat73
import scipy.io as spio
import numpy as np 
from pathlib import Path
import nibabel as nb
import io as fio
import snirf 
from scipy.spatial import distance
from snirf import Snirf
import snirf as snirf
import h5py

import neuro_dot as ndot


def check_keys(dict):
    '''
    CHECK_KEYS Checks if entries in dictionary are mat-objects. If entries are mat objects, todict is called to change them to nested dictionaries
    '''

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = ndot.todict(dict[key])
    return dict     


def loadmat(filename):
    '''
    LOADMAT Loads files with the *.mat extension.
    
    Function written by 'mergen' on Stack Overflow:
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. 
    
    Loadmat calls the function check_keys to cure all entries
    which are still mat-objects. 
    
    :NOTE:This function does not work for .mat -v7.3 files. 
    '''

    data = spio.loadmat(filename,struct_as_record=False, squeeze_me=True)
    return ndot.check_keys(data)


def loadmat7p3(filename):
    '''
    LOADMAT7P3 Loads files with the *.mat extension in the "mat 7.3" format.
    
    Function written by 'mergen' on Stack Overflow:
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. 
    
    Loadmat7p3 calls the function check_keys to cure all entries
    which are still mat-objects. 
    
    NOTE:This function is to be used for .mat -v7.3 files only. 
    '''
    data = mat73.loadmat(filename, use_attrdict = True )
    return ndot.check_keys(data)


def LoadVolumetricData(filename, pn, file_type):
    '''
    LOADVOLUMETRICDATA Loads a volumetric data file

    [volume, header] = LOADVOLUMETRICDATA(filename, pn, file_type) loads a
    file specified by "filename", path "pn", and "file_type", and returns
    it in two parts: the raw data file "volume", and the header "header",
    containing a number of key-value pairs in 4dfp format.

    [volume, header] = LOADVOLUMETRICDATA(filename) supports a full
    filename input, as long as the extension is included in the file name
    and matches a supported file type.
    
    Supported File Types/Extensions: '.4dfp' 4dfp, 'nii' NIFTI.
    
    NOTE: This function uses the NIFTI_Reader toolbox available on MATLAB
    Central. This toolbox has been included with NeuroDOT 2.
    
    Dependencies: READ_4DFP_HEADER, READ_NIFTI_HEADER, MAKE_NATIVESPACE_4DFP.
    
    See Also: SAVEVOLUMETRICDATA.
    ''' 
    header = dict()
        
    if file_type == '4dfp':
        header_out = ndot.Read_4dfp_Header(filename + '.4dfp.ifh', pn)
        # Read .img file.
        pn = pn + '/' + filename + '.4dfp.img'
        with fio.open(pn,'rb',) as fp:
            volume = np.fromfile(fp, dtype = '<f')
            fp.close()

    ## Put header into native space if not already.
        header_out = ndot.Make_NativeSpace_4dfp(header_out)
        ## Format for output.
        volume =np.squeeze(np.reshape(volume, (int(header_out['nVx']), int(header_out['nVy']), int(header_out['nVz']), int(header_out['nVt']))))
        
        if header_out['acq'] == 'transverse':
            volume = np.flip(volume, 1)
        elif header_out['acq'] == 'coronal':
                volume = np.flip(volume, 1)
                volume = np.flip(volume, 2)
        elif header_out['acq'] == 'sagittal':
                volume = np.flip(volume, 0)
                volume = np.flip(volume, 1)
                volume = np.flip(volume, 2)
    
    elif np.logical_or(np.logical_or((file_type == 'nifti'),  (file_type == 'nii')), (file_type == 'nii.gz')):
        ## Call NIFTI_Reader function.
        ## note: When passing file types, if you have the ".nii" file
        ## extension, you must use that as both the "ext" input AND add it
        ## as an extension on the "filename" input.
        if file_type == 'nifti':
            file_type = 'nii'

        nii = nb.load(pn + filename +  '.' + file_type)
        if nii.header['sform_code'] == 'scanner':
            nii.header['sform_code'] = 1
        elif nii.header['sform_code'] == 'talairach':
            nii.header['sform_code'] = 3
        elif nii.header['sform_code'] == 'unknown':
            nii.header['sform_code'] = 2
        elif nii.header['sform_code'] == 0:
            nii.header['sform_code'] = 0

        if nii.header['qform_code'] == 'scanner':
            nii.header['qform_code'] = 1
        elif nii.header['qform_code'] == 'talairach':
            nii.header['qform_code'] = 3
        elif nii.header['qform_code'] == 'unknown':
            nii.header['qform_code'] = 2
        elif nii.header['qform_code'] == 0:
            nii.header['qform_code'] = 0
            
        volume = np.flip(nii.dataobj,0)
        img_out, header_out = ndot.nifti_4dfp(nii.header, volume, '4') # Convert nifti format header to 4dfp format
        
        header_out['original_header'] = nii.header

        # Convert NIFTI header to NeuroDOT style info metadata
        header_out['format'] = 'NIfTI1' # Nifti1Header class from NiBabel
        header_out['version_of_keys'] = '3.3' 
        header_out['conversion_program'] = 'NeuroDOT_LoadVolumetricData'
        header_out['filename'] = filename + '.nii'
        header_out['bytes_per_pixel'] = nii.header['bitpix']/8
        header_out['nVx'] = header_out['matrix_size'][0]
        header_out['nVy'] = header_out['matrix_size'][1]
        header_out['nVz'] = header_out['matrix_size'][2]
        header_out['nVt'] = header_out['matrix_size'][3]
        header_out['mmx'] = abs(header_out['mmppix'][0])
        header_out['mmy'] = abs(header_out['mmppix'][1])
        header_out['mmz'] = abs(header_out['mmppix'][2])

        orientation = header_out['orientation']

        if orientation == '2':
            header['acq'] = 'transverse'
        elif orientation == '3':
            header['acq'] = 'coronal'
        elif orientation == '4':
            header['acq'] = 'sagittal'

    return volume, header_out


def Make_NativeSpace_4dfp(header_in):
    '''
    MAKE_NATIVESPACE_4DFP Calculates native space for an incomplete 4dfp header.
    
    header_out = MAKE_NATIVESPACE_4DFP(header_in) checks whether
    "header_in" contains the "mmppix" and "center" fields (these are not
    always present in 4dfp files). If either is absent, a default called
    the "native space" is calculated from the other fields of the volume.
    
    See Also: LOADVOLUMETRICDATA, SAVEVOLUMETRICDATA.
    '''
    ## Parameters and Initialization.
    header_out = header_in
    ## Check for mmppix, calculate if not present
    keys = header_out.keys()
    if 'mmppix' not in keys:
        if 'mmx' not in keys or 'mmy' not in keys or 'mmz' not in keys:
            print('*** Error: Required field(s) ''mmx'', ''mmy'', or ''mmz'' not present in input header. ***')
        header_out['mmppix'] = [header_out['mmx'], -header_out['mmy'], -header_out['mmz']]
    ## Check for center, calculate if not present
    if 'center' not in keys:
        if 'nVx' not in keys or 'nVy' not in keys or 'nVz' not in keys:
            print('*** Error: Required field(s) ''nVx'', ''nVy'', or ''nVz'' not present in input header. ***')
        header_out['center'] = np.multiply(header_out['mmppix'], 
                                        np.add(np.array([int(header_out['nVx']), int(header_out['nVy']), int(header_out['nVz'])])/2 ,
                                        [0, 1, 1]))
    return header_out


def nifti_4dfp(header_in, img_in, mode):
    '''
    NIFTI_4DFP Converts between nifti and 4dfp volume and header formats.

    Input:
        :header_in: 'struct' of either a nifti or 4dfp header
        :mode: flag to indicate direction of conversion between nifti and 4dfp
            formats 
                - 'n': 4dfp to Nifti
                - '4': Nifti to 4dfp
    Output:
        :header_out: header in Nifti or 4dfp format
    '''
    if mode == 'n':
        # Initialize default parameters
        AXES_4DFP = [0,1,2,3]
        order = AXES_4DFP
        f_offset = 0
        isFloat = 1
        isSigned = 1
        bperpix = 4
        number_of_dimensions = 4
        timestep = 1
        haveCenter = 1
        scale = 1.0
        offset = 0.0

        # Check the orientation
        if 'acq' in header_in:
            if header_in['acq'] == 'transverse':
                header_in['orientation'] = 2
            elif header_in['acq'] == 'sagittal':
                header_in['orienation'] = 4
        
        # Initialize sform matrix to zeros (3x4)
        sform = np.zeros((3,4), dtype = float)
        for i in range(0,3): #0,1,2
            for j in range(0,4):
                sform[i,j] = 0.0
        
         

        # Adjust order of dimensions depending on the orientation
        if 'orientation' in header_in:
            if 'acq' in header_in:
                if header_in['acq'] == 'coronal':
                    header_in['orientation'] = 3

        if header_in['orientation'] == 4:
            tempv = order[1]
            order[1] = order[0]
            order[0] = tempv
        elif header_in['orientation'] == 3:
            tempv = order[2]
            order[2] = order[1]
            order[1] = tempv

         # Assign diagonal of sform to ones
        for i in range(0,3):
            sform[order[i],i] = 1.0     
        dims = number_of_dimensions
        dim = np.zeros((4,1))
        if 'nVx' in header_in:
            header_in['matrix_size'] = [header_in['nVx'], header_in['nVy'], header_in['nVz'], header_in['nVt']]
        dim[0] = header_in['matrix_size'][0]
        dim[1] = header_in['matrix_size'][1]
        dim[2] = header_in['matrix_size'][2]
        dim[3] = header_in['matrix_size'][3]

        spacing = np.zeros((3,1))
        spacing[0] = header_in['mmppix'][0]
        spacing[1] = header_in['mmppix'][1]
        spacing[2] = header_in['mmppix'][2]

        center = np.zeros((3,1))
        center[0] = header_in['center'][0]
        center[1] = header_in['center'][1]
        center[2] = header_in['center'][2]

        if header_in['orientation'] == 4:
            center[2] = -center[2]
            spacing[2] = -spacing[2]
            center[2] = spacing[2] *(dim[2] +1)-center[2]
        elif header_in['orientation'] == 3:
            center[0] = -center[0]
            spacing[0] = -spacing[0]
            center[0] = spacing[0]*(dim[0] +1) - center[0]

        # Adjust for Fortran 1-indexing
        for i in range(0,3):
            center[i] = center[i] - spacing[i]
            center[i] = -center[i]
        # Add sform to t4trans
        t4trans = np.zeros((4,4))
        for i in range(0,3):
            for j in range(0,4):
                t4trans[i,j] = sform[i,j]
        for i in range(0,3):
            # apply spacing
            for j in range(0,3):
                sform[i,j] = t4trans[i,j]*spacing[j]
            for j in range(0,3):
                sform[i,3] = sform[i,3] + center[j]*t4trans[i,j]

        ## Save Nifti
        # to_lpi
        used = 0
        for i in range(0,2):
            max = -1.0
            k = -1
            for j in range(0,3):
                if np.logical_and((~(used & (1<<j))), abs(sform[j,i]) > max):
                    max = abs(sform[j,i])
                    k = j
            used = used | (1<<k)
            order[i] = k

        if used == 3:
            order[2] = 2
        elif used == 5:
            order[2] = 1
        elif used == 6:
            order[2] = 0

        orientation = 0

        for i in range(0,3):
            if sform[order[i],i] < 0.0:
                orientation = orientation ^ (1<<i)
        # auto_orient_header 
        for i in range(0,3):
            # Flip axes
            if orientation & (1<<i):
                for j in range(0,3):
                    sform[j,3] = (dim[i]-1)*sform[j,i] + sform[j,3]
                    sform[j,i] = -sform[j,i]
        orig_sform = sform
        auto_orient_sform = np.zeros((3,4))
        # Re order axes to x, y, z, t
        for i in range(0,3):
            for j in range(0,3):
                sform[i,order[j]] = sform[i,j]

        # Load it back into the sform
        for i in range(0,3):
            for j in range(0,3):
                sform[i,j] = sform[i,j]
        # Define revorder
        revorder = np.zeros((4,1)) # Line 425
        for i in range(0,4):            # Line 426
            revorder[order[i]] = i

        spacing = [0.0,0.0,0.0] # Initialize spacing
        for i in range(0,3):
            for j in range(0,3):
                spacing[i] = spacing[i] + sform[j,i]*sform[j,i]
            spacing[i] = math.sqrt(spacing[i])
        # Create output Nifti-style header
        header_out = nb.Nifti1Header()
        header_out['dim'] = [4, 
            dim[int(revorder[0])],
            dim[int(revorder[1])], 
            dim[int(revorder[2])],
            dim[int(revorder[3])],0,0,0]
        header_out['pixdim'] = [1.0,
            spacing[0],
            spacing[1], 
            spacing[2], 
            timestep, 
            0,0,0]
        header_out['srow_x'] = [sform[0,0], sform[0,1],
            sform[0,2], sform[0,3]]
        header_out['srow_y'] = [sform[1,0], sform[1,1],
            sform[1,2], sform[1,3]]
        header_out['srow_z'] = [sform[2,0], sform[2,1],
            sform[2,2], sform[2,3]]  
        header_out['sform_code'] = 3
        header_out['qform_code'] = 3
        header_out['sizeof_hdr'] = 348
        header_out['aux_file'] = ''
        header_out['descrip'] = header_in['filename'] + '.4dfp.ifh converted with nifti_4dfp'
        header_out['vox_offset'] = 352
        NIFTI_UNITS_MM = 2
        NIFTI_UNITS_SEC = 8
        header_out['xyzt_units'] = NIFTI_UNITS_MM + NIFTI_UNITS_SEC
        header_out['datatype'] = 16
        header_out['bitpix'] = 32

        ## Adjust Volume
        outmem = np.zeros(np.shape(img_in))

        ## auto_orient
        nan_found =0
        i = 0
        val_flip = np.zeros((1,4))
        in_val = np.zeros((4,1))
        out_val = np.zeros((4,1))
        target_length = np.zeros((4,1))
        in_length = np.shape(img_in)
        voxels = img_in
        rData = voxels
        for i in range(1, len(in_length)+1):
            target_length[order(i-1)+1] = in_length(i-1)
            val_flip[i-1] = orientation & (1 <<(i-1))

        # Flip
        if header_in['orientation'] == 2:
            val_flip = np.zeros((1,4))
        
        if header_out['dim'][4] <= 1:
            val_flip = val_flip[0:2]
        
        for k in range(1, len(val_flip)+1):
            if np.any(orig_sform[0:2,k-1] < 0):
                if k != 4:
                    img_in = np.flip(img_in, k-1)
        
        # Permute if needed
        new_dim = np.zeros((1,3))
        for i in range(1,4):
            idx = np.where(target_length[i-1] == dim[0:2])
            if len(idx) > 1:
                idx = idx[0]
            dim[idx] = 0
            new_dim[i-1] = idx
        
        val_flip_new = np.zeros(np.shape(val_flip))
        if not np.array_equal(new_dim, [1,2,3]):
            img_in = np.transpose(img_in,new_dim)
            val_flip_new[:] = val_flip[new_dim]
            val_flip_combined = np.logical_and(val_flip, val_flip_new)
            idx2 = np.where(val_flip_combined == 1)
            if header_in['acq'] == 'coronal':
                img_in = np.flip(img_in, 0)
                img_in = np.flip(img_in, 1)
                img_in = np.flip(img_in, 2)
            else:
                if idx2 != 4:
                    img_in = np.flip(img_in, idx2)
        
        idx_flip = np.find(val_flip > 0)
        if (idx_flip > 0).any():
            for i in range(1, len(idx_flip + 1)):
                if idx_flip != 4:
                    img_xfm = np.flip(img_in, idx_flip[i-1])
        else:
            img_xfm = img_in
        img_xfm = np.flip(img_xfm, 0)
        img_out = img_xfm


    elif mode == '4':
    #  Convert from Nifti to 4dfp style header

    #  Look for the raw structure- if the raw is passed in, do as
    #  normal; if the whole header, then look for
    #  header_in.raw.
        if 'raw' in header_in:
            header_in = header_in['raw']
    #  Initialize parameters
        AXES_NII = [0,1,2,3] # From common-format.h 
        axes = AXES_NII # nifti_format.c:342
        if header_in['sform_code'] == 'scanner':
            header_in['sform_code'] = 1
        elif header_in['sform_code'] == 'talairach':
            header_in['sform_code'] = 3
        elif header_in['sform_code'] == 'unknown':
            header_in['sform_code'] = 2
        elif header_in['sform_code'] == 0:
            header_in['sform_code'] = 0

        if header_in['qform_code'] == 'scanner':
            header_in['qform_code'] = 1
        elif header_in['qform_code'] == 'talairach':
            header_in['qform_code'] = 3
        elif header_in['qform_code'] == 'unknown':
            header_in['qform_code'] = 2
        elif header_in['qform_code'] == 0:
            header_in['qform_code'] = 0

        # Parse_nifti
        order = axes # nifti_format.c:344
        sform = np.zeros((4,4))
        for i in range(0,3):  # nifti_format.c:371
            for j in range(0,3): # nifti_format.c:372
                sform[i,j] = 0.0 # nifti_format.c:373

        if header_in['sform_code']!= 'unknown':  # nifti_format.c:377
            for i in range(0,4): # nifti_format.c:379
                sform[0,i] = header_in['srow_x'][i] # nifti_format.c:381
                sform[1,i] = header_in['srow_y'][i] # nifti_format.c:382
                sform[2,i] = header_in['srow_z'][i] # nifti_format.c:383
        else:
            if header_in['qform_code'] != 'unknown': # nifti_format.c:406
                b = header_in['quatern_b']
                c = header_in['quatern_c']
                d = header_in['quatern_d']
                a = math.sqrt(1.0 - (b**2 + c**2 + d**2))
                # generate rotation matrix (Sform)
                sform[0,0] = a**2 + b**2 - c**2 - d**2
                sform[0,1] = 2*b*c - 2*a*d
                sform[0,2] = 2*b*d + 2*a*c
                sform[1,0] = 2*b*c + 2*a*d
                sform[1,1] = a**2 + c**2 - b**2 - d**2
                sform[1,2] = 2*c*d - 2*a*b
                sform[2,0] = 2*b*d - 2*a*c
                sform[2,1] = 2*c*d + 2*a*b
                sform[2,2] = a**2 + d**2 - c**2 - b**2

                if header_in['pixdim'][0] < 0.0:
                    header_in['pixdim'][3] = -header_in['pixdim'][3] # /* read nifti1.h:1005 for yourself, i can't make this stuff up */
            
                for j in range(0,3):
                    sform[j,0] = sform[j,0]*header_in['pixdim'][1]
                    sform[j,1] = sform[j,1]*header_in['pixdim'][2]
                    sform[j,2] = sform[j,2]*header_in['pixdim'][3]

                sform[0,3] = header_in['qoffset_x']
                sform[1,3] = header_in['qoffset_y']
                sform[2,3] = header_in['qoffset_z']
            else: #  do it the originless way
                sform[0,0] = header_in['pixdim'][1]
                sform[1,1] = header_in['pixdim'][2]
                sform[2,2] = header_in['pixdim'][3]

        # to_lpi
        used = 0
        for i in range(0,2):
            max = -1.0
            k = -1
            for j in range(0,3):
                if np.logical_and((~(used & (1<<j))), abs(sform[j,i]) > max):
                    max = abs(sform[j,i])
                    k = j
            used = used | (1<<k)
            order[i] = k
        if used == 3:
            order[2] = 2
        elif used == 5:
            order[2] = 1
        elif used == 6:
            order[2] = 0

        orientation = 0


        for i in range(0,3):
            if sform[order[i],i] < 0.0:
                orientation = orientation ^ (1<<i)


        revorder = np.zeros((4,1)) # 4dfp_format.c:425
        for i in range(0,4): # 4dfp_format.c:426
            revorder[order[i]] = i

        # auto_orient_header
        orientation = orientation ^ (1 << int(revorder[0]))
        orientation = orientation ^ (1 << int(revorder[1]))
        temp_sform = np.zeros((3,4))
        orig_sform = sform
        for i in range(0,3):
            # Flip axes
            if (orientation & (1<<i)):
                for j in range(0,3):
                    sform[j,3] = (header_in['dim'][i+1]-1)*sform[j,i] + sform[j,3]
                    sform[j,i] = -sform[j,i]

        # Re-order axes to x, y, z, t
        for i in range(0,3):
            for j in range(0,3):
                sform[i,order[j]] = sform[i,j]

        # Load it back into the sform
        for i in range(0,3):
            for j in range(0,2):
                sform[i,j] = sform[i,j]
    
        # Initialize spacing
        spacing = [0,0,0,1]
        for i in range(0,3):
            for j in range(0,3):
                spacing[i] = spacing[i] + sform[j,i]**2
            spacing[i] = math.sqrt(spacing[i])
        
        spacing[0] = -spacing[0] # keep the +, -, - convention in the .ifh, x and z are flipped later */
        spacing[1] = -spacing[1] # we do this here to specify we want the flips to take place before the t4 transform is applied */

        # Initialize t4trans
        t4trans = nm.repmat([0.0], 4,4) # 4dfp_format.c:453
        t4trans[3,3] = 1.0 # 4dfp_format.c:456
        
        # First, invert sform to get t4
        for i in range(0,3): # 4dfp_format.c:460
            for j in range(0,3): # 4dfp_format.c:462
                sform[i,j] = sform[i,j]/spacing[j] # 4dfp_format.c:464 

        determinant = 0
        for i in range(0,3): # 4dfp_format.c:467-477
            # Determinant
            temp = 1.0
            temp2 = 1.0
            for j in range(0,3):
                temp = temp*sform[j,(i+j) % 3]
                temp2 = temp2*sform[j,(i-j+3) % 3]
            determinant = determinant + temp-temp2
            temp = 1.0

        # Adjugate
        t4trans = nm.repmat([0.0], 4,4) # Line 4dfp_format.c:453
        t4trans[3,3] = 1.0 # 4dfp_format.c:456
        for i in range(0,3): # 4dfp_format.c:480-488
            a = (i+1) % 3
            b = (i +2) % 3
            for j in range(0,3):
                c = (j+1) % 3
                d = (j+2) % 3
                t4trans[j,i] =(sform[a,c]*sform[b,d]) - (sform[a,d] *  sform[b,c])
        # Divide t4trans by determinant
        t4trans[0:3,0:3] = t4trans[0:3,0:3]/determinant  

        # Initialize center and assign values from sform multiplied by
        # t4trans (4dfp_format.c:497-505)
        center = [0,0,0]
        for i in range(0,3):
            for j in range(0,3):
                center[i] = center[i] + (sform[j,3]*t4trans[i,j])
        ## Center lines 4dfp_format.c:513-518
        for i in range(0,3):
            center[i] = -center[i]
            center[i] = center[i] + spacing[i]

        ## Center 4dfp_format.c:522-527
        center[0] = (spacing[0] * (header_in['dim'][int(revorder[0])+1]+1))-center[0]
        center[0] = -center[0]
        spacing[0] = -spacing[0]

        center[2] = (spacing[2] * (header_in['dim'][int(revorder[2])+1]+1))-center[2]
        center[2] = -center[2]
        spacing[2] = -spacing[2]

        header_out = dict()
        header_out['matrix_size'] = [header_in['dim'][int(revorder[0])+1],
            header_in['dim'][int(revorder[1])+1],
            header_in['dim'][int(revorder[2])+1],
            header_in['dim'][int(revorder[3])]+1]
        header_out['acq'] = 'transverse'
        header_out['nDim'] = 4
        header_out['orientation'] = 2
        header_out['scaling_factor'] = np.absolute(spacing)
        header_out['mmppix'] = [spacing[0], spacing[1], spacing[2]]
        header_out['center'] = [center[0], center[1], center[2]]

        ## Adjust Volume
        outmem = np.zeros(np.shape(img_in))

        ## auto_orient
        nan_found =0
        i = 0
        val_flip = np.zeros((1,4))
        in_val = np.zeros((4,1))
        out_val = np.zeros((4,1))
        target_length = np.zeros((4,1))
        in_length = np.shape(img_in)
        voxels = img_in
        rData = voxels
        for i in range(1, len(in_length)+1):
            target_length[order(i-1)+1] = in_length(i-1)
            val_flip[i-1] = orientation & (1 <<(i-1))

        # Flip
        if header_in['orientation'] == 2:
            val_flip = np.zeros((4,1))
        
        idx_flip = np.find(val_flip > 1)
        new_order = range(1,len(in_length))
        if (idx_flip > 0).any():
            idx_new = np.flip(idx_flip)
            new_order[idx_new] = np.flip(new_order[idx_new])
            img_xfm = np.transpose(img_in, new_order)
        else:
            img_xfm = img_in
        for k in range(1, len(val_flip) + 1):
            if np.logical_and((orig_sform[k-1,0:2] < 0).any(), orig_sform[k-1,3] > 0):
                img_xfm = np.flip(img_xfm, k-1)
        img_out = img_xfm
    return img_out, header_out


def ndot2nirs(data, info):
    """
    Translation from NeuroDOT-compatible format to .nirs
    This function takes in data and an info structure in NeuroDOT format
    and converts it to the .nirs format.

    Inputs:
        'data': time series data arranged by [#channels x #samples]
        'info': NeuroDOT formatted info structure

    Outputs:
        'nirsData': .nirs formatted data structure containing the following
            'd': time series data arranged by [#time points x #channels]
            'SD': structure containing the source and detector
            information in addition to the data measurement list
            't': time point array
            's': time points and stimulus onsets
            'ml': the data measurement list
    """
    
    # Initialize nirsData
    nirsData = {}
    
    ## Data
    nirsData['d'] = data.T

    ## SD
    nirsData['SD'] = {}
    if 'io' in info:
        nirsData['SD']['nDets'] = info['io']['Nd']
        nirsData['SD']['nSrcs'] = info['io']['Ns']
    else:
        nirsData['SD']['nDets'] = info['optodes']['dpos3'].shape[0]
        nirsData['SD']['nSrcs'] = info['optodes']['spos3'].shape[0]
        
    nirsData['SD']['Lambda'] = np.unique(info['pairs']['lambda']).tolist()

    ## Time 't'
    if 'misc' in info and 'startTime' in info['misc'] and info['misc']['startTime'] == 0:
        nirsData['t'] = (np.arange(nirsData['d'].shape[0]) / info['system']['framerate']).tolist()
    else:
        nirsData['t'] = (np.arange(1, nirsData['d'].shape[0] + 1) / info['system']['framerate']).tolist()

    ## Stimulus 's'
    if 'paradigm' in info:
        fields = list(info['paradigm'].keys())
        pulses = [field for field in fields if len(field) >= 5 and field.startswith('Pulse')]
        pulses.sort()
        num_synchs = len(pulses)
        nirsData['s'] = np.zeros((len(nirsData['t']), num_synchs))
        
        for j, pulse in enumerate(pulses):
            pulseidx = []
            if isinstance(info['paradigm'][pulse], int):
                info['paradigm'][pulse] = [info['paradigm'][pulse]]
            for synch in info['paradigm'][pulse]:
                nearest_idx = np.argmin(np.abs(np.array(nirsData['t']) - (info['paradigm']['synchpts'][synch-1] / info['system']['framerate'])))
                pulseidx.append(nearest_idx)
            nirsData['s'][pulseidx, j] = 1

    ## Optodes
    if 'spos2' in info['optodes']:
        nirsData['SD']['SrcPos2'] = info['optodes']['spos2']
    if 'spos3' in info['optodes']:
        nirsData['SD']['SrcPos3'] = info['optodes']['spos3']
    if 'dpos2' in info['optodes']:
        nirsData['SD']['DetPos2'] = info['optodes']['dpos2']
    if 'dpos3' in info['optodes']:
        nirsData['SD']['DetPos3'] = info['optodes']['dpos3']

    ## MeasList
    nirsData['SD']['MeasList'] = np.column_stack((info['pairs']['Src'], 
                                                  info['pairs']['Det'], 
                                                  np.ones(len(info['pairs']['Src']), dtype=int), 
                                                  info['pairs']['WL']))

    nirsData['ml'] = nirsData['SD']['MeasList']

    ## Aux
    if 'misc' in info and 'aux' in info['misc']:
        nirsData['aux'] = info['misc']['aux']
    return nirsData


def ndot2snirf(filename, output=None, full=0, type='snirf'):
    # Parameters and Initialization
    if output is None:
        output = filename.split('.')[0]
    
    # Load NeuroDOT file
    try:
        with h5py.File(filename, 'r') as f:
            v73 = 1
    except OSError:
        v73 = 0
    if v73 == 0:
        data = ndot.loadmat(filename)['data']     
        info = ndot.loadmat(filename)['info']    
    if v73 == 1:
        data = ndot.loadmat7p3(filename)['data']     
        info = ndot.loadmat7p3(filename)['info']    

    # Create SNIRF Data and Metadata Structures
    if type.lower() == 'snirf':
        snf = Snirf()
        snf.nirs.appendGroup()
        snf.nirs[0].conversionProgram = 'ndot2snirf'
        if data is not None:
            snf.nirs[0].data.appendGroup()
            snf.nirs[0].data[0].dataTimeSeries = data.T
        if info is not None:
            if info['system']['framerate']:
                snf.nirs[0].metaDataTags.framerate = float(info['system']['framerate'])
            if info['system']['init_framerate']:
                snf.nirs[0].metaDataTags.init_framerate = float(info['system']['init_framerate'])
            if 'io' in info:
                if 'FrequencyUnit' in info['io']:
                    snf.nirs[0].metaDataTags.FrequencyUnit = str(info['io']['FrequencyUnit'])
                else:
                    snf.nirs[0].metaDataTags.FrequencyUnit = 'Hz'
                
                if 'TimeUnit' in info['io']:
                    snf.nirs[0].metaDataTags.TimeUnit = str(info['io']['TimeUnit'])
                else:
                    snf.nirs[0].metaDataTags.TimeUnit = 's'
                if 'LengthUnit' in info['io']:
                    snf.nirs[0].metaDataTags.LengthUnit = str(info['io']['LengthUnit'])
                else:
                    snf.nirs[0].metaDataTags.LengthUnit = 'mm'
                if 'ManufacturerName' in info['io']:
                    snf.nirs[0].metaDataTags.ManufacturerName = str(info['io']['ManufacturerName'])
                else:
                    snf.nirs[0].metaDataTags.ManufacturerName = 'n/a'
                if 'Model' in info['io']:
                    snf.nirs[0].metaDataTags.Model = str(info['io']['Model'])
                else:
                    snf.nirs[0].metaDataTags.Model = 'n/a'
                if 'SubjectID' in info['io']:
                    snf.nirs[0].metaDataTags.SubjectID = str(info['io']['SubjectID'])
                else:
                    snf.nirs[0].metaDataTags.SubjectID = 'n/a'
                if 'MeasurementDate' in info['io']:
                    snf.nirs[0].metaDataTags.MeasurementDate = str(info['io']['MeasurementDate'])
                else:
                    snf.nirs[0].metaDataTags.MeasurementDate = 'n/a'
                if 'MeasurementTime' in info['io']:
                    snf.nirs[0].metaDataTags.MeasurementTime = str(info['io']['MeasurementTime'])
                else:
                    snf.nirs[0].metaDataTags.MeasurementTime = 'n/a'
                if 'unix_time' in info['io']:  
                    snf.nirs[0].metaDataTags.UnixTime = str(info['io']['unix_time'])
                else:
                    snf.nirs[0].metaDataTags.UnixTime = 'n/a'
                if 'Nt' in info['io']:
                    snf.nirs[0].metaDataTags.Nt = int(info['io']['Nt'])
                else:
                    snf.nirs[0].metaDataTags.Nt = 0
                if 'Nd' in info['io']:
                    snf.nirs[0].metaDataTags.Nd = int(info['io']['Nd'])
                else:
                    snf.nirs[0].metaDataTags.Nd = len(info['optodes']['dpos3'])
                if 'Ns' in info['io']:
                    snf.nirs[0].metaDataTags.Ns = int(info['io']['Ns'])
                else:
                    snf.nirs[0].metaDataTags.Ns = len(info['optodes']['spos3'])
                if 'Nwl' in  info['io']:
                    snf.nirs[0].metaDataTags.Nwl = int(info['io']['Nwl'])
                else:
                    snf.nirs[0].metaDataTags.Nwl = max(info['pairs']['lambda'])
            
            T = 1/info['system']['framerate']
            nTp = snf.nirs[0].data[0].dataTimeSeries.shape[0]
            snf.nirs[0].data[0].time = np.arange(1, nTp+1) * T

            # Set time, length, and frequency units
            if 'misc' in info:
                if 'time' in info['misc'] and info['misc']['time'] is not None:
                    snf.nirs[0].data[0].time = info['misc']['time']
                if 'TimeUnit' in info['misc'] and info['misc']['TimeUnit'] is not None:
                    snf.nirs[0].metaDataTags.TimeUnit = info['misc']['TimeUnit']
                if 'LengthUnit' in info['misc'] and info['misc']['LengthUnit'] is not None:
                    snf.nirs[0].metaDataTags.LengthUnit = info['misc']['LengthUnit']
                if 'FrequencyUnit' in info['misc'] and info['misc']['FrequencyUnit'] is not None:
                    snf.nirs[0].metaDataTags.FrequencyUnit = info['misc']['FrequencyUnit']                    
                if 'time' in info['misc']:
                    snf.nirs[0].data[0].time = info['misc']['time']
                if 'TimeUnit' in info['misc']:
                    snf.nirs[0].metaDataTags.TimeUnit = info['misc']['TimeUnit']
                if 'LengthUnit' in info['misc']:
                    snf.nirs[0].metaDataTags.LengthUnit = info['misc']['LengthUnit']
                if 'FrequencyUnit' in info['misc']:
                    snf.nirs[0].metaDataTags.FrequencyUnit = info['misc']['FrequencyUnit']
                snf.nirs[0].aux.appendGroup()
                snf.nirs[0].aux[0].name = 0
                snf.nirs[0].aux[0].dataTimeSeries = 0
                snf.nirs[0].aux[0].time = 0
                snf.nirs[0].aux[0].time = 0
                snf.nirs[0].aux[0].timeOffset = 0
                
            
            if 'io' in info:
                if 'comment' in info['io']:
                    snf.nirs[0].metaDataTags.comment = info['io']['comment']
                if 'a' in info['io']:
                    snf.nirs[0].metaDataTags.MeasurementDate = info['io']['a']['date']
                elif 'date' in info['io']:
                    snf.nirs[0].metaDataTags.MeasurementDate = info['io']['date']
                else:
                    snf.nirs[0].metaDataTags.MeasurementDate = 'n/a'
                if 'enc' in info['io']:
                    snf.nirs[0].metaDataTags.enc = info['io']['enc']
                if 'framesize' in info['io']:
                    snf.nirs[0].metaDataTags.framesize = info['io']['framesize']
                if 'naux' in info['io']:
                    snf.nirs[0].metaDataTags.naux = info['io']['naux']
                if 'nblank' in info['io']:
                    snf.nirs[0].metaDataTags.nblank = info['io']['nblank']
                if 'nframe' in info['io']:
                    snf.nirs[0].metaDataTags.nframe = info['io']['nframe']
                if 'nmotu' in info['io']:
                    snf.nirs[0].metaDataTags.nmotu = info['io']['nmotu']
                if 'nsamp' in info['io']:
                    snf.nirs[0].metaDataTags.nsamp = info['io']['nsamp']
                if 'nts' in info['io']:
                    snf.nirs[0].metaDataTags.nts = info['io']['nts']
                if 'pad' in info['io']:
                    snf.nirs[0].metaDataTags.pad = info['io']['pad']
                if 'run' in info['io']:
                    snf.nirs[0].metaDataTags.run = info['io']['run']
                if 'a' in info['io'] and 'time' in info['io']['a']:
                    snf.nirs[0].metaDataTags.MeasurementTime = info['io']['a']['time']
                elif 'time' in info['io']:
                    snf.nirs[0].metaDataTags.MeasurementTime = info['io']['time']
                if 'tag' in info['io']: 
                    snf.nirs[0].metaDataTags.tag = info['io']['tag']

        # info.optodes
        if 'optodes' in info:
            if 'CapName' in info['optodes']:
                snf.nirs[0].metaDataTags.CapName = info['optodes']['CapName']
            if 'dpos2' in info['optodes']:
                snf.nirs[0].probe.detectorPos2D = info['optodes']['dpos2']
            if 'dpos3' in info['optodes']:
                snf.nirs[0].probe.detectorPos3D = info['optodes']['dpos3']
            if 'spos2' in info['optodes']:
                snf.nirs[0].probe.sourcePos2D = info['optodes']['spos2']
            if 'spos3' in info['optodes']:
                snf.nirs[0].probe.sourcePos3D = info['optodes']['spos3']
        
        # info.pairs
        if 'pairs' in info:
            pairs = info['pairs']
            if 'Src' in pairs:
                snf.nirs[0].data[0].measurementList.sourceIndex = pairs['Src']
            if 'Det' in pairs:
                snf.nirs[0].data[0].measurementList.detectorIndex = pairs['Det']
            if 'WL' in pairs:
                snf.nirs[0].data[0].measurementList.wavelengthIndex = pairs['WL']
            if 'Mod' in pairs:
                snf.nirs[0].data[0].measurementList.Mod = pairs['Mod']
            if 'lambda' in pairs:
                snf.nirs[0].data[0].measurementList.wavelengthActual = pairs['lambda']
                snf.nirs[0].probe.wavelengths = list(set(pairs['lambda']))
        
        # info.system
        if not 'system' in info:
            info['system'] = {'framerate': None, 'init_framerate': None, 'Padname': None}
        if 'framerate' in info['system']:
            T = 1 / info['system']['framerate']
            nTp = snf.nirs[0].data[0].dataTimeSeries.shape[0]
            timeArray = np.arange(1, nTp+1) * T
            snf.nirs[0].data.time = timeArray
        else:
            snf.nirs[0].data.time = np.arange(1, len(data)+1) * 1
        
        if 'PadName' in info['system']:
            snf.nirs[0].metaDataTags.PadName = info['system']['PadName']

        # info.paradigm
        if 'paradigm' in info:
            fields = [field for field in info['paradigm']]
            idxPulse = [field.startswith('Pulse_') for field in fields]
            pulses = [fields[i] for i in range(len(fields)) if idxPulse[i]]
            k = 0
            if 'misc' in info:
                if 'stimDuration' in info['misc']:
                    stim_duration = info['misc']['stimDuration']
                else:
                    stim_duration = np.tile(1, [len(pulses),1])
            if 'stim_duration' not in locals():
                if 'Pulse_2' in info['paradigm'] and 'Pulse_1' in info['paradigm'] and 'Pulse_3' not in info['paradigm']:
                    difference = np.around(np.diff(info['paradigm']['synchpts'])).astype(int)
                    difference = difference[1:len(difference)]
                    temp_1 = (info['paradigm']['Pulse_1'][1:]).astype(int) 
                    temp_1 = temp_1-2
                    stim_duration = [round(difference[0] / info['system']['framerate']), round(np.mean(difference[temp_1-1]) / info['system']['framerate'])]
                else:
                    difference = np.diff(info['paradigm']['synchpts'])
                    stim_duration = []
                    for i in range(len(pulses)):
                        stim_duration.append(round(np.mean(difference) / info['system']['framerate']))
            for i in range(len(pulses)):
                k += 1
                if 'synchtype' not in info['paradigm']:
                    info['paradigm']['synchtype'] = np.zeros(len(info['paradigm']['synchpts']))
                    for j in range(len(pulses)):
                        info['paradigm']['synchtype'][info['paradigm'][pulses[j]]-1] = j
                    info['paradigm']['synchtype'] = info['paradigm']['synchtype'].T
                if pulses[i] in info['paradigm']:
                    synchtimes = info['paradigm']['synchpts'] / info['system']['framerate']
                    np_ = np.shape([(info['paradigm'][pulses[i]])-1])[-1]
                    snf.nirs[0].stim.appendGroup()
                    snf.nirs[0].stim[i].data = np.vstack([[np.array(synchtimes[info['paradigm'][pulses[i]]-1]).astype(int)-1], np.ones(np_) * stim_duration[i],[(info['paradigm']['synchtype'][info['paradigm'][pulses[i]]-1]).astype(int)-1]]).T
                    snf.nirs[0].stim[i].name= str(i)
                if 'io' in info:
                    if 'a' in info['io'] and info['io']['a']['tag'].lower() == 'resta':
                        info['io']['a']['tag'] = 'rest'     
        
        snf.nirs[0].data[0].measurementList.dataTypeIndex = np.zeros(data.shape[0], dtype=int)
        snf.nirs[0].data[0].measurementList.dataType = np.ones(data.shape[0], dtype=int)
    # Save Output .snirf file
    outfname = output 
    print('Saving to: ', outfname)
    snf.save(f"{output}.snirf")
    return snf


def nirs2ndot(filename, save_file=1, output=None):
    '''
    NIRS2NDOT reads in a .nirs file and converts it to NeuroDOT
    Compatible variables: data and info
    
    Input:
        :filename: the full file name including extension

        :data: the NIRS data in the format of: N_meas x N_samples

        :info: the data structure that holds information pertaining to data acquisition

        :save_file: can be set to 0 to suppress saving data and info variables to a .mat file
        
        :output: the name of the output file to save the data and info variables to. If not specified, the output file will be the same as the input file name without the extension.
    '''
    # Parameters and Initialization
    if output is None:
        output = filename.rsplit('.', 1)[0]

    # Data
    temp_file =  filename
    try:
        nirsData = ndot.loadmat7p3(temp_file)
    except Exception:
        nirsData = ndot.loadmat(temp_file)

    data = nirsData['d'].T

    # IO and System parameters
    info = {
        'io': {
            'Nd': nirsData['SD']['nDets'],
            'Ns': nirsData['SD']['nSrcs'],
            'Nwl': len(nirsData['SD']['Lambda'])*1.0,
            'nframe': len(nirsData['t'])*1.0
        },
        'system': {
            'framerate': 1 / np.mean(np.diff(nirsData['t']),dtype=float)
        }
    }
    info['misc'] = {'startTime': float(0)} if nirsData['t'][0] == 0 else {'startTime': float(1)}

    # Paradigm
    if np.ndim(nirsData['s']) == 1:
        num_stim = 1
    else:
        num_stim = nirsData['s'].shape[1]
    num_synchs = np.sum(nirsData['s'] == 1)
    field_names = []

    if num_synchs > 0:
        if num_stim == 1:
             synchs = [np.where(nirsData['s'][:] == 1)[0] for j in range(num_stim)]
        else:
            synchs = [np.where(nirsData['s'][:, j-1] == 1)[0] for j in range(num_stim)]
        synchTot = np.sort(np.concatenate(synchs))
        synchTot_dbl = np.sort(np.concatenate(synchs)).astype(dtype=float)+1.0
        info['paradigm'] = {'synchpts': synchTot_dbl.reshape(len(synchTot),1).astype(dtype=float), 'synchtype': np.zeros((len(synchTot),1), dtype=float)}
        
        for k, synch in enumerate(synchs):
            field_name = f'Pulse_{k + 1}'
            field_names.append(field_name)
            temp = np.flatnonzero(np.isin(synchTot, synch))
            info['paradigm'][field_name] = temp.reshape(len(temp),1)+1.0
            for idx in info['paradigm'][field_name]:
                info['paradigm']['synchtype'][int(idx)-1] = k + 1

        info['paradigm']['synchtimes'] = (synchTot_dbl  / info['system']['framerate']).reshape(len(synchTot),1)

    # Optodes
    spos3, dpos3, spos2, dpos2 = None, None, None, None
    if 'SrcPos' in nirsData['SD'] and 'DetPos' in nirsData['SD']:
        dimension = nirsData['SD']['SrcPos'].shape[1]
        
        if dimension == 3:  # if 3D coords, get 3D pos, then check for 2D-specific pos
            spos3, dpos3 = nirsData['SD']['SrcPos'], nirsData['SD']['DetPos']

            if 'SrcPos2' in nirsData['SD'] and 'DetPos2' in nirsData['SD']:
                spos2, dpos2 = nirsData['SD']['SrcPos2'], nirsData['SD']['DetPos2']
            else:
                print('No 2D optode positions in NIRS data, please create your own 2D layout.')
                print('Place the 2D source coordinates in info.optodes.spos2 and the 2D detector coordinates in info.optodes.dpos2.')

        elif dimension == 2:    # if 2D coords, check for 3D-specific pos first
            if 'SrcPos3' in nirsData['SD'] and 'DetPos3' in nirsData['SD']:
                spos3, dpos3 = nirsData['SD']['SrcPos3'], nirsData['SD']['DetPos3']
                spos2, dpos2 = nirsData['SD']['SrcPos'], nirsData['SD']['DetPos']
            else:
                print('No 3D optode coordinates in this nirs data.')
                print('Please find the 3D SD coordinates and place the source coordinates in info.optodes.spos3 and the detector coordinates in info.optodes.dpos3.')
                print('You may need to make a neuroDOT pad file and/or head model to do this.')
        else:
            print('No optode coordinates in this nirs data.')
            print('Please find the 3D SD coordinates and place the source coordinates in info.optodes.spos3 and the detector coordinates in info.optodes.dpos3.')
            print('You may need to make a neuroDOT pad file and/or head model to do this.')
            print('No 2D optode positions in NIRS data, please create your own 2D layout.')
            print('Place the 2D source coordinates in info.optodes.spos2 and the 2D detector coordinates in info.optodes.dpos2.')
            
    elif 'SrcPos3' in nirsData['SD'] and 'DetPos3' in nirsData['SD']:
        spos3, dpos3 = nirsData['SD']['SrcPos3'], nirsData['SD']['DetPos3']
        if 'SrcPos2' in nirsData['SD'] and 'DetPos2' in nirsData['SD']:
            spos2, dpos2 = nirsData['SD']['SrcPos2'], nirsData['SD']['DetPos2']
        else:
            print('No 2D optode positions in NIRS data, please create your own 2D layout.')
            print('Place the 2D source coordinates in info.optodes.spos2 and the 2D detector coordinates in info.optodes.dpos2.')
    else:
        print('No optode coordinates in this nirs data.')
        print('Please find the 3D SD coordinates and place the source coordinates in info.optodes.spos3 and the detector coordinates in info.optodes.dpos3.')
        print('You may need to make a neuroDOT pad file and/or head model to do this.')
        print('No 2D optode positions in NIRS data, please create your own 2D layout.')
        print('Place the 2D source coordinates in info.optodes.spos2 and the 2D detector coordinates in info.optodes.dpos2.')

    # Variables for generating info.optodes and info.pairs
    lambda_ = nirsData['SD']['Lambda']
    SD_sep = np.array([distance.euclidean(spos3[int(nirsData['SD']['MeasList'][ii][0]-1),:],
                                         dpos3[int(nirsData['SD']['MeasList'][ii][1]-1),:])
                      for ii in range(nirsData['SD']['MeasList'].shape[0])])

    avg_SD_sep = np.mean(np.abs(SD_sep))
    min_SD_sep = np.min(np.abs(SD_sep))

    if ((avg_SD_sep >= 10) and (avg_SD_sep < 100)) or ((min_SD_sep >= 1) and (min_SD_sep < 100)):
        mult = 1
    elif (avg_SD_sep >= 1) and (avg_SD_sep < 10) or ((min_SD_sep >= 0.1) and (min_SD_sep < 10)):
        mult = 10
    elif (avg_SD_sep >= 0.1) and (avg_SD_sep < 1) or ((min_SD_sep >= 0.01) and (min_SD_sep < 1)):
        mult = 100
    elif (avg_SD_sep >= 0.01) and (avg_SD_sep < 0.1) or ((min_SD_sep >= 0.001) and (min_SD_sep < 0.1)):
        mult = 1000
    else:
        print('Optode position units are larger than meters, data is wonky, please fix your data')

    spos3 = spos3 * mult
    dpos3 = dpos3 * mult

    if spos3.shape[1] > spos3.shape[0]:
        spos3 = spos3.T
        dpos3 = dpos3.T
        if spos2 is not None:
            spos2 = spos2.T
            dpos2 = dpos2.T

    info['optodes'] = {
        'spos3': spos3,
        'dpos3': dpos3,
        'plot3orientation': {'i': 'R2L', 'j': 'P2A', 'k': 'D2V'}
    }

    # Pairs
    lambdaArray = np.array([lambda_[int(nirsData['SD']['MeasList'][ii, 3]) - 1] for ii in range(nirsData['SD']['MeasList'].shape[0])])
    r3dArray = SD_sep

    info['pairs'] = {
        'Src': nirsData['SD']['MeasList'][:, 0].reshape(nirsData['SD']['MeasList'].shape[0],1),
        'Det': nirsData['SD']['MeasList'][:, 1].reshape(nirsData['SD']['MeasList'].shape[0],1),
        'NN': nirsData['SD']['MeasList'][:, 2].reshape(nirsData['SD']['MeasList'].shape[0],1),
        'WL': nirsData['SD']['MeasList'][:, 3].reshape(nirsData['SD']['MeasList'].shape[0],1),
        'Mod': ['CW'] * nirsData['SD']['MeasList'][:,3].shape[0],
        'r3d': r3dArray.reshape(r3dArray.shape[0],1),
        'lambda': lambdaArray.reshape(lambdaArray.shape[0],1)
    }

    info = ndot.calc_NN(info, 10)
    info['pairs']['NN'] = info['pairs']['NN'].reshape(nirsData['SD']['MeasList'].shape[0],1)

    info['tissue'] = {
        'affine': np.eye(4),
        'affine_target': 'MNI'
    }

    if spos2 is not None:
        spos2 = spos2 * mult
        dpos2 = dpos2 * mult

        r2dArray = [distance.euclidean(spos2[nirsData['SD']['MeasList'][ii, 0] - 1], 
                                        dpos2[nirsData['SD']['MeasList'][ii, 1] - 1])
                    for ii in range(nirsData['SD']['MeasList'].shape[0])]

        info['optodes']['spos2'] = spos2
        info['optodes']['dpos2'] = dpos2

        info['pairs']['r2d'] = r2dArray.reshape(r3dArray.shape[0],1)
    # else:
    #     info['pairs']['r2d'] = r3dArray.reshape(r3dArray.shape[0],1)

    if save_file == 1:
        outputfilename = output
        ndot.savemat(outputfilename, {'data': data, 'info': info})

    return data, info


def Read_4dfp_Header(filename, pn):
    '''
    READ_4DFP_HEADER Reads the .ifh header of a 4dfp file.
    
    header = READ_4DFP_HEADER(filename, pn) reads an .ifh text file specified
    by "filename", containing a number of key-value pairs. The specific
    pairs are parsed and stored as fields of the output structure "header".
        
    See Also: LOADVOLUMETRICDATA.
    '''
    ## Parameters and initialization.
    header_ifh = {}
    header  ={}
    ## Open file.
    fid = open(filename, 'rb')
    path = Path(pn + filename)
    assert path.exists()
    ## Read text.    
    with path.open() as fp:
        for line in fp:
            token = line.split(":=")
            if len(token) != 2:
                continue
            key = token[0].strip()
            value = token[1].strip()
            header_ifh[key] =  value

    ## Interpret key-value pairs
    keys = header_ifh.keys()
    if 'version of keys' in keys:
        header['version_of_keys'] = header_ifh['version of keys']

    if 'number format' in keys:
        header['format'] = header_ifh['number format']

    if 'conversion program' in keys:
        header['conversion_program'] = header_ifh['conversion program']

    if 'name of data file' in keys:
        header['filename'] = header_ifh['name of data file']

    if 'number of bytes per pixel' in keys:
        header['bytes_per_pixel'] = header_ifh['number of bytes per pixel']

    if 'imagedata byte order' in keys:
        if header_ifh['imagedata byte order'] =='bigendian':
            header['byte'] = 'b'
        elif header_ifh['imagedata byte order'] =='littleendian':
            header['byte'] = 'l'
        else:
            header['byte'] = 'b'
    else:
        header['byte'] = 'b'

    if 'orientation' in keys:
        if header_ifh['orientation'] == '2':
            header['acq'] = 'transverse'
        elif header_ifh['orientation'] == '3':
            header['acq'] = 'coronal'
        elif header_ifh['orientation'] == '4':   
            header['acq'] = 'sagittal'

    if 'number of dimensions' in keys:
        header['nDim'] = header_ifh['number of dimensions']

    if 'matrix size [1]' in keys:
        header['nVx'] = header_ifh['matrix size [1]']

    if 'matrix size [2]' in keys:
        header['nVy'] = header_ifh['matrix size [2]']

    if 'matrix size [3]' in keys:
        header['nVz'] = header_ifh['matrix size [3]']

    if 'matrix size [4]' in keys:
        header['nVt'] = header_ifh['matrix size [4]']

    if 'scaling factor (mm/pixel) [1]' in keys:
        header['mmx'] = float(header_ifh['scaling factor (mm/pixel) [1]'])

    if 'scaling factor (mm/pixel) [2]' in keys:
        header['mmy'] = float(header_ifh['scaling factor (mm/pixel) [2]'])

    if 'scaling factor (mm/pixel) [3]' in keys:
        header['mmz'] = float(header_ifh['scaling factor (mm/pixel) [3]'])

    if 'patient ID' in keys:
        header['subjcode'] = header_ifh['patient ID']

    if 'date' in keys:
        header['filedate'] = header_ifh['date'] 

    if 'mmppix' in keys:    
        header['mmppix'] = header_ifh['mmppix'].split()
        header['mmppix'] = list(map(float, header['mmppix']))

    if 'center' in keys:
        header['center'] = header_ifh['center'].split()
        header['center'] = list(map(float, header['center']))

    return header

def savemat(filename, data):
    p = filename + '.mat'
    spio.savemat(p, data)


def SaveVolumetricData(volume, header, filename, pn, file_type):
    '''
    SAVEVOLUMETRICDATA Saves a volumetric data file.

    SAVEVOLUMETRICDATA(volume, header, filename, pn, file_type) saves
    volumetric data defined by "volume" and "header" into a file specified
    by "filename", path "pn", and "file_type".

    SAVEVOLUMETRICDATA(volume, header, filename) supports a full
    filename input, as long as the extension is included in the file name
    and matches a supported file type.

        - Supported File Types/Extensions: '.4dfp' 4dfp, '.nii' NIFTI.

    Dependencies: WRITE_4DFP_HEADER, NIFTI_4DFP.

    See Also: LOADVOLUMETRICDATA, MAKE_NATIVESPACE_4DFP.
    '''

    if file_type.lower() == '4dfp':
        if header['nVt']!=np.size(volume,3):
            print('Warning: Stated header 4th dimension size ' + 
            str(header['nVt']) + ' does not equal the size of the volume ' +
            str(np.size(volume,3)) + '. Updating header info.')
            header['nVt']=np.size(volume,3)

        ## Write 4dfp header.
        if header['acq'] == 'transverse':
            volume = np.flip(volume,1)
        elif header['acq'] == 'coronal':
            volume = np.flip(volume, 1)
            volume = np.flip(volume, 2)
        elif header['acq'] == 'sagittal':
            volume = np.flip(volume, 0)
            volume = np.flip(volume, 1)
            volume = np.flip(volume, 2)
                
        ndot.Write_4dfp_Header(header, pn + filename)
    
        ## Write 4dfp image file.
        volume = np.squeeze(np.reshape(volume, header['nVx'] * header['nVy'] * header['nVz'] * header['nVt'], 1))
        # Write .img file
        imgfname = filename + '.4dfp.img'
        with open(filename + '.4dfp.img','wb') as fp:
            np.array(volume.shape).tofile(fp)
            volume.T.tofile(fp)

    elif np.logical_or(file_type.lower() == 'nifti', file_type.lower() == 'nii'):
        volume = np.flip(volume, 0) # Convert back from LAS to RAS for NIFTI.      
        
        if 'original_header' in header: # if header was loaded as nii, and you are saving it as nifti again
            nii = header['original_header']
            # Required fields
            if 'Description' in nii:
                if len(nii['Description']) > 80:    
                    nii['Description'] = 'Converted with NeuroDOT nifti_4dfp'
            affine=[nii['srow_x'],nii['srow_y'], nii['srow_z'],[0,0,0,1]]
            nifti_image = nb.Nifti1Image(volume, affine, nii)
            nb.save(nifti_image, pn + '/' + filename) 

            
        else: # Build nii header from information in NeuroDOT 4dfp header
            new_vol, nifti_header = ndot.nifti_4dfp(header,volume, 'n')
            affine=[nifti_header['srow_x'],nifti_header['srow_y'], nifti_header['srow_z'],[0,0,0,1]]
            nifti_image = nb.Nifti1Image(new_vol, affine , nifti_header)
            nb.save(nifti_image, pn + '/' + filename+'.nii') 
            

def snirf2ndot(filename, pn, save_file =0, output = None, dtype = []):
    '''
    SNIRF2NDOT takes a file with the 'snirf' extension in the SNIRF format and converts it to NeuroDOT formatting.

    This function depends on the "Snirf" class from pysnirf2.

    Inputs:
        :Filename: the name of the file to be converted, followed by the .snirf extension.
        :Save_file: flag which determines whether the data will be saved to a 'mat' file in NeuroDOT format. (default = 1) 
        :Output: the filename (without extension) of the .mat file to be saved.
        :Type: optional - the only currently acceptable value is "snirf"
    Outputs:
        :data: NeuroDOT formatted data (# of channels x # of samples)
        :info: NeuroDOT formatted metadata "info"
    '''
    if pn is None:
        pn = '/'
    
    if dtype is None:
        dtype = 'snirf'
    if output is None:
        output = filename
    if save_file is None:
        save_file = 0
    fn = pn + filename
    snf = snirf.Snirf(fn, 'r')
    info = dict()
    custom_io = ['Nd', 'Ns','Nwl','comment','enc', 'framesize', 'naux','nblank','nframe',
                'nmotu','nsamp','nts','pad','run','tag','Nt','PadName']
    if snf['original_header']:
        info['original_header'] = snf.original_header

    if snf.nirs[0].data[0]:
        info['system'] = dict()
        data = np.transpose(snf.nirs[0].data[0].dataTimeSeries)
        if hasattr(snf.nirs[0].data[0],'time'):
            info['system']['framerate'] = 1/np.mean(np.diff(snf.nirs[0].data[0].time))
            info['system']['init_framerate'] = info['system']['framerate']
        if snf.nirs[0].metaDataTags.TimeUnit == 'ms':
            info['system']['framerate'] = info['system']['framerate']*1e3
            info['system']['init_framerate'] = info['system']['framerate']*1e3
    info['io'] = dict()
    if 'original_header' in snf:
        info['original_header'] = snf.original_header
        if 'io' in snf.original_header:
            if 'a' in snf.original_header.io:
                info['io']['a'] = snf.original_header.io.a
                info['io']['b'] = snf.original_header.io.b
            else:
                info['io'] = snf.original_header.io

    if snf:
        if snf['original_header']:
            if snf.original_header.io:
                if snf.original_header.io.a:
                    info['io']['a'] = snf.original_header.io.a
                    info['io']['b'] = snf.original_header.io.b
                else:
                    info['io'] = snf.original_header.io     
        else:
            if snf.nirs[0].metaDataTags:
                if not ('io' in info):
                    info['io'] = dict()
                info['misc'] = dict()
                info['misc']['metaDataTags'] = snf.nirs[0].metaDataTags
                info['misc']['time'] = snf.nirs[0].data[0].time
                if hasattr(snf.nirs[0].metaDataTags,'framerate'):
                    info['system']['framerate'] = snf.nirs[0].metaDataTags.framerate
                    info['system']['init_framerate'] = info['system']['framerate']
                if hasattr(snf.nirs[0].metaDataTags,'Nd'):
                    info['io']['Nd'] = snf.nirs[0].metaDataTags.Nd
                else:
                    info['io']['Nd'] = len(snf.nirs[0].probe.detectorPos3D)

                if hasattr(snf.nirs[0].metaDataTags,'Ns'):
                    info['io']['Ns'] = snf.nirs[0].metaDataTags.Ns
                else:
                    info['io']['Ns'] = len(snf.nirs[0].probe.sourcePos3D)

                if hasattr(snf.nirs[0].metaDataTags,'MeasurementDate'):
                    info['io']['date'] = str(snf.nirs[0].metaDataTags.MeasurementDate)  # this might need to have join() because in Matlab the measurement date was a character array
                if hasattr(snf.nirs[0].metaDataTags, 'MeasurementTime'):
                    info['io']['time'] = str(snf.nirs[0].metaDataTags.MeasurementTime) # this might need to have join() because in Matlab the measurement date was a character array
                if hasattr(snf.nirs[0].metaDataTags, 'UnixTime'):
                    info['io']['unix_time'] = snf.nirs[0].metaDataTags.UnixTime
                if hasattr(snf.nirs[0].metaDataTags,'Nwl'):
                    info['io']['Nwl'] = snf.nirs[0].metaDataTags.Nwl
                else:
                    info['io']['Nwl'] = len(snf.nirs[0].probe.wavelengths)
                if hasattr(snf.nirs[0].metaDataTags,'comment'):
                    info['io']['comment'] = snf.nirs[0].metaDataTags.comment
                if hasattr(snf.nirs[0].metaDataTags,'enc'):
                    info['io']['enc'] = snf.nirs[0].metaDataTags.enc
                if hasattr(snf.nirs[0].metaDataTags,'framesize'):
                    info['io']['framesize'] = snf.nirs[0].metaDataTags.framesize
                if hasattr(snf.nirs[0].metaDataTags,'naux'):
                    info['io']['naux'] = snf.nirs[0].metaDataTags.naux

    if snf.nirs[0].probe:
        info['optodes'] = dict()
        if 'CapName' in snf.nirs[0].metaDataTags:
            info['optodes']['CapName'] = snf.nirs[0].metaDataTags.CapName
        if 'detectorPos2D' in snf.nirs[0].probe:
            info['optodes']['dpos2'] = snf.nirs[0].probe.detectorPos2D
        if 'detectorPos3D' in snf.nirs[0].probe:
            info['optodes']['dpos3'] = snf.nirs[0].probe.detectorPos3D
        if 'sourcePos2D' in snf.nirs[0].probe:
            info['optodes']['spos2'] = snf.nirs[0].probe.sourcePos2D
        if 'sourcePos3D' in snf.nirs[0].probe:
            info['optodes']['spos3'] = snf.nirs[0].probe.sourcePos3D
            
    if snf.nirs[0].data[0].measurementList:  
        info['pairs'] = dict()
        info['pairs']['Src'] = np.zeros(len(snf.nirs[0].data[0].measurementList[:]))
        info['pairs']['Det'] = np.zeros(len(snf.nirs[0].data[0].measurementList[:]))
        info['pairs']['WL'] = np.zeros(len(snf.nirs[0].data[0].measurementList[:]))
        info['pairs']['lambda'] = np.zeros(len(snf.nirs[0].data[0].measurementList[:]))     
        for j in range(0, len(snf.nirs[0].data[0].measurementList[:])):
            if hasattr(snf.nirs[0].data[0].measurementList[0], 'sourceIndex'):
                info['pairs']['Src'][j] = snf.nirs[0].data[0].measurementList[j].sourceIndex
            if hasattr(snf.nirs[0].data[0].measurementList[0], 'detectorIndex'):
                info['pairs']['Det'][j] = snf.nirs[0].data[0].measurementList[j].detectorIndex
            if hasattr(snf.nirs[0].data[0].measurementList[0], 'wavelengthActual'):
                info['pairs']['lambda'][j] = snf.nirs[0].data[0].measurementList[j].wavelengthActual
            if hasattr(snf.nirs[0].data[0].measurementList[0],'wavelengthIndex'):
                info['pairs']['WL'][j] = snf.nirs[0].data[0].measurementList[j].wavelengthIndex
        if not hasattr(snf.nirs[0].data[0].measurementList[0],'wavelengthActual'):
            info['pairs']['lambda'] = list()
            if 'wavelengths' in snf.nirs[0].probe:
                wavelengths = snf.nirs[0].probe.wavelengths
            for k in range(1, len(wavelengths)+1):
                info['pairs']['lambda'][np.where(info['pairs']['WL'] == k)[1]] = wavelengths[k]                 
            if np.size(info['pairs']['lambda'],1) > 1:
                info['pairs']['lambda'] = np.transpose(info['pairs']['lambda'])
                
        gridTemp = dict()
        if not hasattr(snf, 'original_header'):
            gridTemp['spos3']=snf.nirs[0].probe.sourcePos3D
            gridTemp['dpos3']=snf.nirs[0].probe.detectorPos3D
            #Enforce that arrays are column-wise
            if np.size(gridTemp['spos3'],1) > np.size(gridTemp['spos3'],0):
                gridTemp['spos3'] = np.transpose(gridTemp['spos3'])
                gridTemp['dpos3'] = np.transpose(gridTemp['dpos3'])
            
            if hasattr(snf.nirs[0].probe, 'sourcePos2D') and hasattr(snf.nirs[0].probe,'detectorPos2D'):
                if  not (snf.nirs[0].probe.sourcePos2D is None):
                    if snf.nirs[0].probe.sourcePos2D.all() != None and snf.nirs[0].probe.detectorPos2D.all() != None:
                        gridTemp['spos2']=snf.nirs[0].probe.sourcePos2D
                        gridTemp['dpos2']=snf.nirs[0].probe.detectorPos2D
                    #Enforce that arrays are column-wise
                        if np.size(gridTemp['spos3'],1) > np.size(gridTemp['spos3'],0):
                            gridTemp['spos2'] = np.transpose(gridTemp['spos2'])
                            gridTemp['dpos2'] = np.transpose(gridTemp['dpos2'])
            params = dict()
            params['lambda']= snf.nirs[0].probe.wavelengths
            tempInfo=ndot.Generate_pad_from_grid(gridTemp,params, info)          
            data_measList = np.array(np.c_[info['pairs']['Src'],info['pairs']['Det'],info['pairs']['WL']])
            full_measList =np.array(np.c_[tempInfo['pairs']['Src'],tempInfo['pairs']['Det'],tempInfo['pairs']['WL']])
            idxmeaslist = [np.where((full_measList == data_meas).all(axis=1))[0] for data_meas in data_measList]
            info['pairs']['Mod'] = tempInfo['pairs']['Mod'][idxmeaslist,:]
            info['pairs']['r3d']=tempInfo['pairs']['r3d'][idxmeaslist]
            info['pairs']['r2d'] = tempInfo['pairs']['r2d'][idxmeaslist]
            info['pairs']['NN'] = tempInfo['pairs']['NN'][idxmeaslist]
            info['pairs']['lambda'] = tempInfo['pairs']['lambda'][idxmeaslist]
        
            avg_r3d = np.mean(info['pairs']['r3d'])
            if (avg_r3d >=1) & (avg_r3d <=10):# Changed max_log to min_log 2/1/23
                mult = 10
            elif (avg_r3d >=0.1) & (avg_r3d <=1):
                mult = 100
            elif (avg_r3d >=0) & (avg_r3d <=0.1):
                mult = 1000
            else:
                mult = 1
            info['optodes']['spos3'] =np.multiply(gridTemp['spos3'],mult)
            info['optodes']['dpos3'] = np.multiply(gridTemp['dpos3'],mult)
            info['pairs']['r3d'] = np.multiply(info['pairs']['r3d'],mult)
            info['pairs']['r2d'] = np.multiply(info['pairs']['r2d'],mult)

        if hasattr(snf, 'original_header'): 
            info['paradigm'] = snf.original_header.paradigm
        else:
            if len(snf.nirs[0].data[0].time) == 2 :# Create Time array if one is not provided
                T = 1/info['system']['framerate']
                nTp = np.shape(snf.nirs[0].data[0].dataTimeSeries)[0]
                timeArray = np.array(range(0, nTp-1)*T)
            else:
                timeArray = np.array(snf.nirs[0].data[0].time)
            if hasattr(snf.nirs[0], 'stim'):
                Total_synchs = []
                Total_synchtypes = []
                Npulses = len(snf.nirs[0].stim)
                for i in range(0, Npulses):
                    Total_synchs =  np.append(Total_synchs,snf.nirs[0].stim[i].data[:,0])
                    Total_synchtypes =  np.append(Total_synchtypes,np.tile([i+1], len(snf.nirs[0].stim[i].data)))
                info['paradigm'] = dict()
                info['paradigm']['synchtimes'] = np.sort(Total_synchs)
                sortedIdx = np.argsort(Total_synchs)
                info['paradigm']['synchtype'] = Total_synchtypes[sortedIdx]
                pulsenum = 0
                for j in range(0,Npulses):
                    pulsenum = pulsenum +1
                    info['misc']['stimDuration']  =np.zeros(len(snf.nirs[0].stim[j].data[:,1]))
                    info['misc']['stimDuration'][j] = snf.nirs[0].stim[j].data[0,1]
                    label = 'Pulse_'+ str(pulsenum)
                    info['paradigm'][label] = np.array(np.where(info['paradigm']['synchtype'] == pulsenum)) + 1
                    info['paradigm'][label] = np.transpose(info['paradigm'][label])
                info['paradigm']['synchpts'] = np.zeros((np.shape(info['paradigm']['synchtimes'])))    
                for j in range(0,len(info['paradigm']['synchtimes'])):
                    info['paradigm']['synchpts'][j] = np.argmin(abs((np.array(timeArray)- info['paradigm']['synchtimes'][j])))
                info['paradigm']['init_synchpts'] = info['paradigm']['synchpts']   
        
        if not hasattr(snf.nirs[0].metaDataTags, 'SubjectID'):
            info['misc']['subject_id'] = 'default' #required snirf field
        else:
            info['misc']['subject_id'] = snf.nirs[0].metaDataTags.SubjectID

    
    # Order info.pairs and data by wavelength, then by detector
    if np.shape(info['pairs']['Src'])[0]> 1:
        info['pairs']['Src'] = np.transpose(info['pairs']['Src'])
    if np.shape(info['pairs']['Det'])[0] > 1:
        info['pairs']['Det'] = np.transpose(info['pairs']['Det'])
    if np.shape( info['pairs']['WL'])[0] > 1:
        info['pairs']['WL'] = np.transpose(info['pairs']['WL'])
    info['pairs']['Src'] = np.reshape(info['pairs']['Src'],(len(info['pairs']['Src']),1))
    info['pairs']['Det'] = np.reshape(info['pairs']['Det'],(len(info['pairs']['Det']),1))
    info['pairs']['WL'] = np.reshape(info['pairs']['WL'],(len(info['pairs']['WL']),1))
    info['pairs']['lambda'] = np.reshape(info['pairs']['lambda'],(len(info['pairs']['lambda']),1))
    info['pairs']['NN'] = np.reshape(info['pairs']['NN'],(len(info['pairs']['NN']),1))
    lexsorter = np.lexsort((info['pairs']['Det'],info['pairs']['WL']), axis = 0)
    info['pairs']['Det'] = info['pairs']['Det'][lexsorter]
    info['pairs']['Src'] = info['pairs']['Src'][lexsorter]
    info['pairs']['WL'] = info['pairs']['WL'][lexsorter]
    info['pairs']['lambda'] = info['pairs']['lambda'][lexsorter]
    info['pairs']['r3d']= info['pairs']['r3d'][lexsorter]
    info['pairs']['r2d'] = info['pairs']['r2d'][lexsorter]
    info['pairs']['NN'] = info['pairs']['NN'][lexsorter]
    info['pairs']['Det'] = np.double(info['pairs']['Det'])
    info['pairs']['Src'] = np.double(info['pairs']['Src'])
    info['pairs']['WL'] = np.double(info['pairs']['WL'])
    info['pairs']['lambda'] = np.double(info['pairs']['lambda'])
    info['pairs']['r3d']= np.double(info['pairs']['r3d'])
    info['pairs']['r2d'] = np.double(info['pairs']['r2d'])
    info['pairs']['NN'] = np.double(info['pairs']['NN'])
    data = np.squeeze(data[lexsorter])
    
# Save Output NeuroDOT File
    if save_file == 1:
        if (output is None) and (pn is None):
            __file__ = filename
        elif pn is None:
            __file__ = output
        else:
            __file__ = pn
        p = os.path.dirname(os.path.realpath(__file__)) # this needs to be changed to be the path to where the data file is, not in the neurodot py dir
        p = p + '\\' + output + '.mat'
        outputs = dict()
        outputs['data'] = data
        outputs['info'] = info
        spio.savemat(p, {'data':data, 'info':info})
    return data, info    


def todict(matobj):
    '''
    TODICT Recursively constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = ndot.todict(elem)
        else:
            dict[strg] = elem
    return dict


def Write_4dfp_Header(header, filename, pn):
    '''
    WRITE_4DFP_HEADER Writes a 4dfp header to a .ifh file.

    WRITE_4DFP_HEADER(header, filename) writes the input "header" in 4dfp
    format to an .ifh file specified by "filename".
    
    See Also: SAVEVOLUMETRICDATA.
    '''
    ## Parameters and Initialization
    ## Read text.    
    os.chdir(pn)
    with open(filename + ".ifh", 'w') as fp:
    ## Print input header to file.
        print('INTERFILE :=\n', file = fp)
        print('version of keys := ' + header['version_of_keys'] +'\n', file = fp)
        print('image modality := dot\n', file = fp)
        print('originating system := Neuro-DOT\n', file = fp)
        print('conversion program := MATLABto4dfp\n', file = fp)
        print('original institution := Washington University\n', file = fp)
        print('number format := ' +  header['format'] + '\n', file = fp)
        print('name of data file := ' + filename + '\n', file = fp)
        print('number of bytes per pixel := ' +  str(header['bytes_per_pixel']) +'\n', file = fp)

        if header['byte'] =='b':
            byte = 'big'
        elif header['byte'] == 'l':
            byte = 'little'
    
        print('imagedata byte order := ' +  byte +  'endian\n', file = fp)

        if header.acq == 'transverse':
                print( 'orientation := 2\n', file = fp)
        elif header.acq =='coronal':
                print( 'orientation := 3\n', file = fp)
        elif header.acq == 'sagittal':
                print( 'orientation := 4\n', file = fp)
        

        print('number of dimensions := ' + str(header['nDim']) +  '\n', file = fp)
        print('matrix size [1] := ' + str(header['nVx']) + '\n', file = fp)
        print('matrix size [2] := ' + str(header['nVy']) + '\n', file = fp)
        print('matrix size [3] := ' + str(header['nVz']) + '\n', file = fp)
        print('matrix size [4] := ' + str(header['nVt']) + '\n', file = fp)
        print('scaling factor (mm/pixel) [1] := ' + str(header['mmx']) + '\n', file = fp)
        print('scaling factor (mm/pixel) [2] := ' + str(header['mmy']) + '\n', file = fp)
        print('scaling factor (mm/pixel) [3] := ' + str(header['mmz']) + '\n', file = fp)
        if 'mmppix' in header:
            print('mmppix := ' + str(header['mmppix']) + '\n', file = fp)
        if 'center' in header:
            print('center := ' + str(header['center']) + '\n', file = fp)

    return 