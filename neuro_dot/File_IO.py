import sys
import math
import os
import mat73
import scipy.io as spio
import numpy as np
from pathlib import Path
import nibabel as nb
import io as fio



class io:

    def _check_keys(dict):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in dict:
            if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
                dict[key] = io._todict(dict[key])
        return dict     


    def loadmat(filename):
        '''
        function written by 'mergen' on Stack Overflow:
        https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects. This function does not work for .mat -v7.3 files. 
        '''
        data = spio.loadmat(filename,struct_as_record=False, squeeze_me=True)
        return io._check_keys(data)


    def loadmat7p3(filename):
        '''
        function written by 'mergen' on Stack Overflow:
        https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
        this function should be called instead of direct spio.loadmat
        as it cures the problem of not properly recovering python dictionaries
        from mat files. It calls the function check keys to cure all entries
        which are still mat-objects. This function is to be used for .mat -v7.3 files only. 
        '''
        data = mat73.loadmat(filename, use_attrdict = True )
        return io._check_keys(data)


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
            header_out = io.Read_4dfp_Header(filename + '.4dfp.ifh', pn)
            # Read .img file.
            pn = pn + '/' + filename + '.4dfp.img'
            with fio.open(pn,'rb',) as fp:
                volume = np.fromfile(fp, dtype = '<f')
                fp.close()

        ## Put header into native space if not already.
            header_out = io.Make_NativeSpace_4dfp(header_out)
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

            header_out = io.nifti_4dfp(nii.header, '4') # Convert nifti format header to 4dfp format
            volume = np.flip(nii.dataobj,0)
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
        MAKE_NATIVESPACE_4DFP Calculates native space of incomplete 4dfp header.
     
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
    

    def nifti_4dfp(header_in, mode):
        '''
        Function to convert between nifti and 4dfp header formats.
        Inputs:
         header_in: 'struct' of either a nifti or 4dfp header
         mode: flag to indicate direction of conversion between nifti and 4dfp
         formats 
             - 'n': 4dfp to Nifti
             - '4': Nifti to 4dfp
        Output:
          header_out: header in Nifti or 4dfp format
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
            
            # Assign diagonal of sform to ones
            for i in range(0,3):
                sform[order[i],i] = 1.0       

            # Adjust order of dimensions depending on the orientation
            if header_in['orientation'] == 4:
                tempv = order[1]
                order[1] = order[0]
                order[0] = tempv
            elif header_in['orientation'] == 3:
                tempv = order[2]
                order[2] = order[1]
                order[1] = tempv

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

        elif mode == '4':
        #  Convert from Nifti to 4dfp style header

        #  Look for the raw structure- if the raw is passed in, do as
        #  normal; if the whole header, then look for
        #  header_in.raw.
            
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

        return header_out


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
    

    def SaveVolumetricData(volume, header, filename, pn, file_type):
        '''
        SAVEVOLUMETRICDATA Saves a volumetric data file.

        SAVEVOLUMETRICDATA(volume, header, filename, pn, file_type) saves
        volumetric data defined by "volume" and "header" into a file specified
        by "filename", path "pn", and "file_type".

        SAVEVOLUMETRICDATA(volume, header, filename) supports a full
        filename input, as long as the extension is included in the file name
        and matches a supported file type.

        Supported File Types/Extensions: '.4dfp' 4dfp, '.nii' NIFTI.

        NOTE: This function uses the NIFTI_Reader toolbox available on MATLAB
        Central. This toolbox has been included with NeuroDOT 2.

        Dependencies: WRITE_4DFP_HEADER, WRITE_NIFTI.

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
                    
            io.Write_4dfp_Header(header, pn + filename)
        
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
                nifti_header = io.nifti_4dfp(header, 'n')
                affine=[nifti_header['srow_x'],nifti_header['srow_y'], nifti_header['srow_z'],[0,0,0,1]]
                nifti_image = nb.Nifti1Image(volume, affine , nifti_header)
                nb.save(nifti_image, pn + '/' + filename+'.nii') 
                

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                dict[strg] = io._todict(elem)
            else:
                dict[strg] = elem
        return dict
    

    def Write_4dfp_Header(header, filename, pn):
        '''
        WRITE_4DFP_HEADER Writes 4dfp header to .ifh file.
 
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