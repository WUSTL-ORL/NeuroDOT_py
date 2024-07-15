:py:mod:`neuro_dot.File_IO`
===========================

.. py:module:: neuro_dot.File_IO


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.File_IO.check_keys
   neuro_dot.File_IO.loadmat
   neuro_dot.File_IO.loadmat7p3
   neuro_dot.File_IO.LoadVolumetricData
   neuro_dot.File_IO.Make_NativeSpace_4dfp
   neuro_dot.File_IO.nifti_4dfp
   neuro_dot.File_IO.Read_4dfp_Header
   neuro_dot.File_IO.SaveVolumetricData
   neuro_dot.File_IO.snirf2ndot
   neuro_dot.File_IO.todict
   neuro_dot.File_IO.Write_4dfp_Header



.. py:function:: check_keys(dict)

   CHECK_KEYS Checks if entries in dictionary are mat-objects. 
   If entries are mat objects, todict is called to change them to nested dictionaries


.. py:function:: loadmat(filename)

   LOADMAT Loads files with the *.mat extension.

   Function written by 'mergen' on Stack Overflow:
   https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

   This function should be called instead of direct spio.loadmat
   as it cures the problem of not properly recovering python dictionaries
   from mat files. 

   Loadmat calls the function check_keys to cure all entries
   which are still mat-objects. 

   NOTE:This function does not work for .mat -v7.3 files. 


.. py:function:: loadmat7p3(filename)

   LOADMAT7P3 Loads files with the *.mat extension in the "mat 7.3" format.

   Function written by 'mergen' on Stack Overflow:
   https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

   This function should be called instead of direct spio.loadmat
   as it cures the problem of not properly recovering python dictionaries
   from mat files. 

   Loadmat7p3 calls the function check_keys to cure all entries
   which are still mat-objects. 

   NOTE:This function is to be used for .mat -v7.3 files only. 


.. py:function:: LoadVolumetricData(filename, pn, file_type)

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


.. py:function:: Make_NativeSpace_4dfp(header_in)

   MAKE_NATIVESPACE_4DFP Calculates native space for an incomplete 4dfp header.

   header_out = MAKE_NATIVESPACE_4DFP(header_in) checks whether
   "header_in" contains the "mmppix" and "center" fields (these are not
   always present in 4dfp files). If either is absent, a default called
   the "native space" is calculated from the other fields of the volume.

   See Also: LOADVOLUMETRICDATA, SAVEVOLUMETRICDATA.


.. py:function:: nifti_4dfp(header_in, img_in, mode)

   NIFTI_4DFP Converts between nifti and 4dfp volume and header formats.

   Input:
       :header_in: 'struct' of either a nifti or 4dfp header
       :mode: flag to indicate direction of conversion between nifti and 4dfp
           formats 
               - 'n': 4dfp to Nifti
               - '4': Nifti to 4dfp
   Output:
       :header_out: header in Nifti or 4dfp format


.. py:function:: Read_4dfp_Header(filename, pn)

   READ_4DFP_HEADER Reads the .ifh header of a 4dfp file.

   header = READ_4DFP_HEADER(filename, pn) reads an .ifh text file specified
   by "filename", containing a number of key-value pairs. The specific
   pairs are parsed and stored as fields of the output structure "header".
       
   See Also: LOADVOLUMETRICDATA.


.. py:function:: SaveVolumetricData(volume, header, filename, pn, file_type)

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


.. py:function:: snirf2ndot(filename, pn, save_file=0, output=None, dtype=[])

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


.. py:function:: todict(matobj)

   TODICT Recursively constructs from matobjects nested dictionaries


.. py:function:: Write_4dfp_Header(header, filename, pn)

   WRITE_4DFP_HEADER Writes a 4dfp header to a .ifh file.

   WRITE_4DFP_HEADER(header, filename) writes the input "header" in 4dfp
   format to an .ifh file specified by "filename".

   See Also: SAVEVOLUMETRICDATA.


