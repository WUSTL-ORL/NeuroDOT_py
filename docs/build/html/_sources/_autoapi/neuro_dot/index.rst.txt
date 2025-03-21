:py:mod:`neuro_dot`
===================

.. py:module:: neuro_dot


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   Analysis/index.rst
   DynamicFilter/index.rst
   File_IO/index.rst
   Light_Modeling/index.rst
   Matlab_Equivalent_Functions/index.rst
   Reconstruction/index.rst
   Spatial_Transforms/index.rst
   Temporal_Transforms/index.rst
   Visualizations/index.rst


Package Contents
----------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.adjust_brain_pos
   neuro_dot.applycmap
   neuro_dot.DrawColoredSynchPoints
   neuro_dot.nlrGrayPlots_220324
   neuro_dot.PlotInterpSurfMesh
   neuro_dot.Plot_RawData_Cap_DQC
   neuro_dot.defineticks
   neuro_dot.PlotFalloffData
   neuro_dot.PlotFalloffLL
   neuro_dot.PlotLRMeshes
   neuro_dot.Plot_RawData_Metrics_I_DQC
   neuro_dot.Plot_RawData_Metrics_II_DQC
   neuro_dot.Plot_RawData_Time_Traces_Overview
   neuro_dot.cylinder
   neuro_dot.PlotCapData
   neuro_dot.PlotCapGoodMeas
   neuro_dot.PlotCapMeanLL
   neuro_dot.PlotCapPhysiologyPower
   neuro_dot.PlotSlices_correct_orientation
   neuro_dot.PlotSlices
   neuro_dot.PlotSlicesTimeTrace
   neuro_dot.PlotTimeTraceData
   neuro_dot.vol2surf_mesh
   neuro_dot.check_keys
   neuro_dot.loadmat
   neuro_dot.loadmat7p3
   neuro_dot.LoadVolumetricData
   neuro_dot.Make_NativeSpace_4dfp
   neuro_dot.nifti_4dfp
   neuro_dot.Read_4dfp_Header
   neuro_dot.SaveVolumetricData
   neuro_dot.snirf2ndot
   neuro_dot.todict
   neuro_dot.Write_4dfp_Header
   neuro_dot.affine3d_img
   neuro_dot.change_space_coords
   neuro_dot.GoodVox2vol
   neuro_dot.rotate_cap
   neuro_dot.rotation_matrix
   neuro_dot.detrend_tts
   neuro_dot.nextpow2
   neuro_dot.fft_tts
   neuro_dot.gethem
   neuro_dot.highpass
   neuro_dot.logmean
   neuro_dot.lowpass
   neuro_dot.regcorr
   neuro_dot.resample_tts
   neuro_dot.rms_py
   neuro_dot.calc_NN
   neuro_dot.Generate_pad_from_grid
   neuro_dot.makeFlatFieldRecon
   neuro_dot.DynamicFilter
   neuro_dot.reconstruct_img
   neuro_dot.smooth_Amat
   neuro_dot.spectroscopy_img
   neuro_dot.Tikhonov_invert_Amat
   neuro_dot.BlockAverage
   neuro_dot.CalcGVTD
   neuro_dot.FindGoodMeas
   neuro_dot.normcND
   neuro_dot.normrND



.. py:function:: adjust_brain_pos(meshL, meshR, params=None)

   ADJUST_BRAIN_POS Repositions mesh orientations for display.

   [Lnodes, Rnodes] = ADJUST_BRAIN_POS(meshL, meshR) takes the left and
   right hemispheric meshes "meshL" and "meshR", respectively, and
   repositions them to the proper perspective for display.

   [Lnodes, Rnodes] = ADJUST_BRAIN_POS(meshL, meshR, params) allows the
   user to specify parameters for plot creation.

   Params:
       :ctx: Defines inflation of mesh. 
           std: standard pial mesh *Default*
           inf: inflated mesh
           vinf: very inflated mesh
           flat: flat mesh
       :orientation: Select orientation of volume. 
           t: transverse *Default*
           s: for sagittal
       :view: Sets the view perspective.
           lat: lateral view *Default*
           post: posterior view
           dorsal: dorsal view

   Dependencies: ROTATE_CAP, ROTATION_MATRIX.

   See Also: PLOTLRMESHES.


.. py:function:: applycmap(overlay, underlay, params)

   APPLYCMAP Performs color mapping and fuses images with anatomical models.

   mapped = APPLYCMAP(overlay) fuses the N-D image array "overlay" with a
   default flat gray background and applies a number of other default
   settings to create a scaled and colormapped N-D x 3 RGB image array
   "mapped".

   mapped = APPLYCMAP(overlay, underlay) fuses the image array "overlay"
   with the anatomical atlas volume input "underlay" as the background.

   mapped = APPLYCMAP(overlay, underlay, params) allows the user to
   specify parameters for plot creation.

   Params:
       :TC: Direct map integer data values to defined color map ("True Color"). Default Value: 0
                                   
       :DR: Dynamic range. Default Value: 1000

       :Scale: Maximum value to which image is scaled. Default Value: 90% max

       :PD:  If PD = 1, Sets the entirety of the colormap to run from zero to the Scale value.
           If PD = 0, Colormap is centered around zero. Default Value: 0

       :BG: Background color, as an RGB triplet. Default Value: [0.5, 0.5, 0.5]

       :Saturation: Sets the transparency of the colors. Size: equal to the size of the data. 
           Must be within range [0, 1]. Default Value: none
       
       :Cmap.P: Colormap for positive data values. Default Value: jet
       :Cmap.N: Colormap for negative data values. Default Value: none
       :Cmap.flipP: Logical, flips the positive colormap. Default Value: 0
       :Cmap.flipN: Logical, flips the negative colormap. Default Value: 0
       
       :Th.P: Value of min threshold to display positive data values. Default Value: 25% max
       :Th.N: Value of max threshold to display negative data values. Default Value: -Th.P

       :returns: mapped, map_out, params   

       See Also: PLOTSLICES, PLOTINTERPSURFMESH, PLOTLRMESHES, PLOTCAPMEANLL.


.. py:function:: DrawColoredSynchPoints(info_in, subplt, SPfr=0)

   DRAWCOLOREDSYNCHPOINTS Draws vertical lines over a plot/imagesc type axis to delineate synchronization points on time traces.
   This function assumes the standard NeuroDOT info structure and paradigm
   format

   Info:
       :info.paradigm: Contains all paradigm timing information
       :info.paradigm.synchpts: Contains sample points corresponding to
           stimulus time points of interest (e.g. onsets, rest, etc.)
       :info.paradigm.synchtype: Optional field that contains a marker to
           distinguish between synchronization types, traditionally as 
           sound boops of differing frequencies, i.e., 25, 30, 35 (Hz).
       :info.paradigm.Pulse_1: Indices of synchpts that denote 'bookends' 
           of data collection as well as resting (or OFF) periods of a given
           stimulus.
       :info.paradigm.Pulse_2: Indices of synchpts that correspond to the
           start of a given stimulus epoch (e.g., the start of a flickering
           checkerboard)
       :info.paradigm.Pulse_3: Indices of synchpts of a 2nd stimulus type
       :info.paradigm.Pulse_4: Indices of synchpts of a 3rd stimulus type

   The 2nd input, SPfr, selects whether or not to adjust synchpoint timing
   based on framerate of data. (Default=0).


.. py:function:: nlrGrayPlots_220324(nlrdata, info, mode='auto')

   NLRGRAYPLOTS_220324 This function generates a gray plot figure for measurement pairs
   using only clean data from wavelength 2. Input data "nlrdata" is assumed
   to be filtered and resampled. The data is grouped into info.pairs.r2d<20,
   20<=info.pairs.r2d<30, and 30<=info.pairs.r2d<40.


.. py:function:: PlotInterpSurfMesh(volume, meshL, meshR, dim, params)

   PLOTINTERPSURFMESH Interpolates volumetric data onto hemispheric meshes for display.

   PLOTINTERPSURFMESH(volume, meshL, meshR, dim) takes functional imaging
   data "volume" and interpolates it onto the left and right hemispheric
   surface meshes given in "meshL" and "meshR", respectively, using the
   spatial information in "dim". The result is a bilateral sagittal view
   of the activations overlain onto the surface of the brain represented
   by the meshes.

   PLOTINTERPSURFMESH(volume, meshL, meshR, dim, params) allows the user to
   specify parameters for plot creation.

   Params:
       :Scale: (90% max) Maximum value to which image is scaled.
       :Th.P: (25% max) Value of min threshold to display positive data values.
       :Th.N: (-Th.P) Value of max threshold to display negative data values.

   Dependencies: VOL2SURF_MESH, PLOTLRMESHES, ADJUST_BRAIN_POS, ROTATE_CAP,
   ROTATION_MATRIX.

   See Also: PLOTSLICES.



.. py:function:: Plot_RawData_Cap_DQC(data, info_in, params=None)

   PLOT_RAWDATA_CAP_DQC Generates plots of cap data quality.
       Includes: Relative average light levels for 2 sets of distances
       of source-detector measurements, the cap good measurements plot, and 
       a measure of the pulse power at each optode location.


.. py:function:: defineticks(start, end, steps, blankLabels=False)

   DEFINETICKS Defines tick marks based on a start point, end point, and step size.

   See: PLOT_RAWDATA_METRICS_I_DQC


.. py:function:: PlotFalloffData(fall_data, separations, params=None, ax=None)

   PLOTFALLOFFDATA A basic falloff plotting function.

   PLOTFALLOFFDATA(data, info) takes one input array "fall_data" and plots
   it against another "separations" to create a falloff plot.

   PLOTFALLOFFDATA(data, info, params) allows the user to specify
   parameters for plot creation.

   h = PLOTFALLOFFDATA(...) passes the handles of the plot line objects
   created.

   Params:
       :fig_size:    [200, 200, 560, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                       If empty, spawns a new figure.
       :xlimits:     'auto'                  Limits of x-axis.
       :xscale:      'linear'                Scaling of x-axis.
       :ylimits:     'auto'                  Limits of y-axis.
       :yscale:      'log'                   Scaling of y-axis.

   See Also: PLOTFALLOFFLL.


.. py:function:: PlotFalloffLL(data, info, params=None, ax=None)

   PLOTFALLOFFLL A light-level falloff visualization.

   PLOTFALLOFFLL(data, info) takes a light-level array "data" of the MEAS
   x TIME format, and generates a plot of each channel's temporal mean
   against its source-detector distance, in the specified groupings.

   PLOTFALLOFFLL(data, info, params) allows the user to specify parameters
   for plot creation.

   Params:
       :fig_size:    [200, 200, 560, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :dimension:   '2D'                    Dimension of pair radii used.
       :rlimits:     (all R2D)               Limits of pair radii displayed.
       :Nnns:        (all NNs)               Number of NNs displayed.
       :Nwls:        (all WLs)               Number of WLs displayed.
       :useGM:       0                       Use Good Measurements.
       :xlimits:     [0, 60]                 Limits of x-axis.
       :xscale:      'linear'                Scaling of x-axis.
       :ylimits:     [1e-6, 1e1]             Limits of y-axis.
       :yscale:      'log'                   Scaling of y-axis.

   Dependencies: PLOTFALLOFFDATA


.. py:function:: PlotLRMeshes(meshL, meshR, params)

   PLOTLRMESHES Renders a pair of hemispheric meshes.

   PLOTLRMESHES(meshL, meshR) renders the data in a pair of left and right
   hemispheric meshes "meshL.data" and "meshR.data", respectively, and
   applies full color mapping to them. If no data is present for either
   mesh, a default gray mesh will be plotted.

   PLOTLRMESHES(meshL, meshR, params) allows the user to
   specify parameters for plot creation.

   Params:
       :fig_size: [20, 200, 960, 420] Default figure position vector.
       :fig_handle: (none) Specifies a figure to target. If empty, spawns a new figure.
       :Scale: (90% max) Maximum value to which image is scaled.
       :PD: 0 Declares that input image is positive definite.
       :cblabels: ([-90% max, 90% max]) Colorbar axis labels. Min defaults to 0 if PD==1,
        both default to +/- Scale if supplied.
       :cbticks: (none) Specifies positions of tick marks on colorbar axis.
       :alpha: (1) Transparency of mesh.
       :view: 'lat' Sets the view perspective.

   Dependencies: ADJUST_BRAIN_POS, ROTATE_CAP, ROTATION_MATRIX, APPLYCMAP.
       
   See Also: PLOTINTERPSURFMESH, VOL2SURF_MESH.


.. py:function:: Plot_RawData_Metrics_I_DQC(data, info, params=None)

   PLOT_RAWDATA_METRICS_I_DQC Generates a single-page data quality report including various metrics of the raw data quality.
   Light fall off as a function of Rsd,
   SNR plots vs. mean light level,
   Source-detector mean light-level plots,
   Power spectra for 830nm at 2 Rsd, and 
   Histogram for measurement noise.


.. py:function:: Plot_RawData_Metrics_II_DQC(data, info, params=None)

   PLOT_RAWDATA_METRICS_II_DQC Generates a single-page data quality report including various metrics of the raw data quality.
       Zoomed raw time traces,
       Light fall off as a function of Rsd,
       Power spectra for 830nm at 2 Rsd, and
       Histogram for measurement noise.


.. py:function:: Plot_RawData_Time_Traces_Overview(data, info_in, params=None)

   PLOT_RAWDATA_TIME_TRACES_OVERVIEW Generates a plot of raw data time traces.
   Time Traces are separated by wavelength (columns). The top row shows 
   time traces for all measurements within a source-detector distance range
   (defaults as 0.1 - 5.0 cm). The bottom row shows the same measurements
   but including only measurements passing a variance threshold (default: 0.075)
   as well as vertical lines corresponding to the stimulus paradigm.


.. py:function:: cylinder(r, h=1, a=0, nt=100, nv=50)

   CYLINDER Parameterizes the cylinder of radius r, height h, and base point a.

   See: PlotCapData


.. py:function:: PlotCapData(SrcRGB, DetRGB, info, fig_axes=None, params=None)

   PLOTCAPDATA A basic plotting function for generating and labeling cap grids.

   PLOTCAPDATA(SrcRGB, DetRGB, info) plots the input RGB information in
   one of three modes:

   Modes:
       :text: - Optode numbers are arranged in a cap grid and colored with the RGB input.
       :patch: - Optodes are plotted as patches and colored with the RGB input.
       :textpatch: - Optodes are plotted as patches and colored with the RGB input, with optode numbers overlain in white.

   PLOTCAPDATA(SrcRGB, DetRGB, info, params) allows the user to specify
   parameters for plot creation.

   Params:
       :fig_size:    [20, 200, 1240, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :dimension:   '2D'                    Specifies either a 2D or 3D
                                           plot rendering.
       :mode:        'textpatch'             Display mode.

   See Also: PLOTCAP, PLOTCAPGOODMEAS, PLOTCAPMEANLL.


.. py:function:: PlotCapGoodMeas(info, fig_axes=None, params=None)

   PLOTCAPGOODMEAS A "Good Measurements" visualization overlaid on a cap grid.

   PLOTCAPGOODMEAS(info) plots a visualization of the Good Measurements
   determined by FINDGOODMEAS and arranges them based on the metadata in
   "info.optodes". Good channels are depicted as green lines, bad channels
   red lines; sources and detectors are given lettering in light blue and
   red.

   The plot title provides tallies for all specified groupings. The next
   line of the title lists how many optodes for which only 33% of their
   measurements are good. These optodes are surrounded with white circles.

   PLOTCAPGOODMEAS(info, params) allows the user to specify parameters for
   plot creation.

   Params:
       :fig_size:    [20, 200, 1240, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :dimension:   '2D'                    Specifies either a 2D or 3D
                                           plot rendering.
       :rlimits:     (all R2D)               Limits of pair radii displayed.
       :Nnns:        (all NNs)               Number of NNs displayed.
       :Nwls:        (all WLs)               Number of WLs averaged and
                                           displayed.
       :mode:        'good'                  Display mode. 'good' displays
                                           channels above noise threhsold,
                                           'bad' below.

   Dependencies: PLOTCAPDATA, ISTABLEVAR.

   See Also: FINDGOODMEAS, PLOTCAP, PLOTCAPMEANLL.


.. py:function:: PlotCapMeanLL(data, info, fig_axes=None, params=None)

   PLOTCAPMEANLL A visualization of mean light levels overlaid on a cap
   grid.

   PLOTCAPMEANLL(data, info) plots an intensity map of the mean light
   levels for specified measurement groupings of each optode on the cap
   and arranges them based on the metadata in "info.optodes".

   PLOTCAPMEANLL(data, info, params) allows the user to specify parameters
   for plot creation.

   Params:
       :fig_size:    [20, 200, 1240, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :dimension:   '2D'                    Specifies either a 2D or 3D
                                           plot rendering.
       :rlimits:     (all R2D)               Limits of pair radii displayed.
       :Nnns:        (all NNs)               Number of NNs displayed.
       :Nwls:        (all WLs)               Number of WLs averaged and
                                           displayed.
       :useGM:       0                       Use Good Measurements.
       :Cmap.P:      'hot'                   Default color mapping.

   Dependencies: PLOTCAPDATA, ISTABLEVAR, APPLYCMAP.

   See Also: PLOTCAP, PLOTCAPGOODMEAS.



.. py:function:: PlotCapPhysiologyPower(data, info, fig_axes=None, params=None)

   PLOTCAPPHYSIOLOGYPOWER A visualization of band limited power OR band-referenced SNR for each optode.


   Params:
       :fig_size:    [20, 200, 1240, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :dimension:   '2D'                    Specifies either a 2D or 3D
                                           plot rendering.
       :rlimits:     (all R2D)               Limits of pair radii displayed.
       :Nnns:        (all NNs)               Number of NNs displayed.
       :Nwls:        (all WLs)               Number of WLs averaged and
                                           displayed.
       :useGM:       0                       Use Good Measurements.
       :Cmap.P:      'hot'                   Default color mapping.

   Dependencies: PLOTCAPDATA, ISTABLEVAR, APPLYCMAP.

   See Also: PLOTCAP, PLOTCAPGOODMEAS.


.. py:function:: PlotSlices_correct_orientation(m)

   PLOTSLICES_CORRECT_ORIENTATION Used in PlotSlices to transpose the dimension, given by "m".


.. py:function:: PlotSlices(underlay, infoVol=None, params=None, overlay=None)

   PLOTSLICES Creates an interactive 3D plot, including transverse, sagittal, and coronal slices.

   PLOTSLICES(underlay) takes a 3D voxel space image "underlay" and
   generates views along the three canonical axes.

   In interactive mode, left-click on any point to move to those slices.
   To reset to the middle of the volume, right-click anywhere. To cancel
   interactive mode, press "Q", "Esc", or the middle mouse button.

   PLOTSLICES(underlay, infoVol) uses the volumetric data in "infoVol" to
   display spatial coordinates of the slices in question.

   PLOTSLICES(underlay, infoVol, params) allows the user to
   specify parameters for plot creation.

   Params:
       :fig_size:    [20, 200, 1240, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :CH:          1                       Turns crosshairs on (1) and off
                                           (0).
       :Scale:       (90% max)               Maximum value to which image is
                                           scaled.
       :PD:          0                       Declares that input image is
                                           positive definite.
       :cbmode:      0                       Specifies whether to use custom
                                           colorbar axis labels.
       :cboff:       0                       If set to 1, no colorbar is
                                           displayed
       :cblabels:    ([-90% max, 90% max])   Colorbar axis labels. When
                                           cbmode==1, min defaults to 0 if
                                           PD==1, both default to +/-
                                           Scale if supplied. When
                                           cbmode==0, then cblabels
                                           dictates colorbar axis limits.
       :cbticks:     (none)                  When cbmode==1, specifies
                                           positions of tick marks on
                                           colorbar axis.
       :slices:      (center frames)         Select which slices are
                                           displayed. If empty, activates
                                           interactive navigation.
       :slices_type: 'idx'                   Use MATLAB indexing ('idx') for
                                           slices, or spatial coordinates
                                           ('coord') as provided by
                                           invoVol.
       :orientation: 't'                     Select orientation of volume.
                                           't' for transverse, 's' for
                                           sagittal.
       
   Note: APPLYCMAP has further options for using "params" to specify
   parameters for the fusion, scaling, and colormapping process.

   PLOTSLICES(underlay, infoVol, params, overlay) overlays the image
   provided by "overlay". When this is done, all color mapping is applied
   to the overlay image, and the underlay is rendered as a grayscale image
   times the RGB triplet in "params.BG".

   Dependencies: APPLYCMAP.

   See Also: PLOTINTERPSURFMESH, PLOTSLICESMOV, PLOTSLICESTIMETRACE.


.. py:function:: PlotSlicesTimeTrace(underlay, infoVol=None, params=None, overlay=None, info=None)

   PLOTSLICESTIMETRACE Creates an interactive 4D plot, including transverse, sagittal, and coronal slices, in addition to a voxel time trace.

   PLOTSLICESTIMETRACE(underlay, infoVol, params, overlay) uses the same
   basic inputs as the function PLOTSLICES to create an interactive 3-axis
   plot of a 4D volume, plus an axis for the time trace of the selected
   voxel.

   PLOTSLICESTIMETRACE(..., info) allows the user to input an "info"
   structure to provide information about the 4D volume's native
   framerate. The default framerate is 1 Hz.

   Params:
       :fig_size:    [20, 200, 840, 720]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :CH:          1                       Turns crosshairs on (1) and off
                                           (0).
       :Scale:       (90% max)               Maximum value to which image is
                                           scaled.
       :PD:          0                       Declares that input image is
                                           positive definite.
       :cbmode:      0                       Specifies whether to use custom
                                           colorbar axis labels.
       :cblabels:    ([-90% max, 90% max])   Colorbar axis labels. When
                                           cbmode==1, min defaults to 0 if
                                           PD==1, both default to +/-
                                           Scale if supplied. When
                                           cbmode==0, then cblabels
                                           dictates colorbar axis limits.
       :cbticks:     (none)                  When cbmode==1, specifies
                                           positions of tick marks on
                                           colorbar axis.
       :slices:      (center frames)         Select which slices are
                                           displayed. If empty, activates
                                           interactive navigation.
       :slices_type: 'idx'                   Use MATLAB indexing ('idx') for
                                           slices, or spatial coordinates
                                           ('coord') as provided by
                                           invoVol.
       :orientation: 't'                     Select orientation of volume.
                                           't' for transverse, 's' for
                                           sagittal.
       :kernel:      [1]                     A sampling kernel for the time
                                           trace plot. Other options:
                                           'gaussian' | 'cube' | 'sphere'.

   Note: APPLYCMAP has further options for using "params" to specify
   parameters for the fusion, scaling, and colormapping process.

   Dependencies: APPLYCMAP.

   See Also: PLOTINTERPSURFMESH, PLOTSLICES.


.. py:function:: PlotTimeTraceData(data, time, params=None, fig_axes=None, coordinates=[1, 1])

   PLOTTIMETRACEDATA A basic time traces plotting function.

   PLOTTIMETRACEDATA(data, time) takes a light-level array "data" of the
   MEAS x TIME format, and plots its time traces.

   PLOTTIMETRACEDATA(data, time, params) allows the user to specify
   parameters for plot creation.

   h = PLOTTIMETRACEDATA(...) passes the handles of the plot line objects
   created.

   Params:
       :fig_size:    [200, 200, 560, 420]    Default figure position vector.
       :fig_handle:  (none)                  Specifies a figure to target.
                                           If empty, spawns a new figure.
       :xlimits:     'auto'                  Limits of x-axis.
       :xscale:      'linear'                Scaling of x-axis.
       :ylimits:     'auto'                  Limits of y-axis.
       :yscale:      'linear'                Scaling of y-axis.

   See Also: PLOTTIMETRACEALLMEAS, PLOTTIMETRACEMEAN.



.. py:function:: vol2surf_mesh(Smesh, volume, dim, params=None)

   VOL2SURF_MESH Interpolates volumetric data onto a surface mesh.

   Smesh = VOL2SURF_MESH(mesh_in, volume, dim) takes the mesh "Smesh"
   and interpolates the values of the volumetric data "volume" at the
   mesh's surface, using the spatial information in "dim". These values
   are output as "Smesh".

   Smesh = VOL2SURF_MESH(Smesh, volume, dim, params) allows the user
   to specify parameters for plot creation.

   Params:
       :OL:      0   If "overlap" data is presented (OL==1), this sets the
                   interpolation method to "nearest". Default is "linear".

   See Also: PLOTINTERPSURFMESH, GOOD_VOX2VOL, AFFINE3D_IMG.


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


.. py:function:: nirs2ndot(filename, save_file=1, output=None)

    NIRS2NDOT reads in a .nirs file and converts it to NeuroDOT
    Compatible variables: data and info
    
    Input:
        :filename: the full file name including extension
        :data: the NIRS data in the format of: N_meas x N_samples
        :info: the data structure that holds information pertaining to data acquisition
        :save_file: can be set to 0 to suppress saving data and info variables to a .mat file
        :output: the name of the output file to save the data and info variables to. If not specified, the output file will be the same as the input file name without the extension.


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


.. py:function:: affine3d_img(imgA, infoA, infoB, affine=np.eye(4), interp_type='nearest')

   AFFINE3D_IMG Transforms a 3D data set to a new space.

   imgB = AFFINE3D_IMG(imgA, infoA, infoB, affine) takes a reconstructed,
   VOX x TIME image "imgA" and transforms it from its initial voxel space
   defined by the structure "infoA" into a target voxel space defined by
   the structure "infoB" and using the transform matrix "affine". The
   output is a VOX x TIME matrix "imgB" in the target voxel space.

   imgB = AFFINE3D_IMG(imgA, infoA, infoB, affine, interp_type) allows the
   user to specify an interpolation method for the INTERP3 function that
   AFFINE3D_IMG uses. Other methods that can be used (input as strings)
   are 'nearest', 'spline', and 'cubic'. The default value is 'linear'.

   See Also: SPECTROSCOPY_IMG, CHANGE_SPACE_COORDS, INTERP3.


.. py:function:: change_space_coords(coord_in, space_info, output_type='coord')

   CHANGE_SPACE_COORDS Applies a look-up to change 3D coordinates into a new space.

   coord_out = CHANGE_SPACE_COORDS(coord_in, space_info, output_type) takes
   a set of coordinates "coord_in" of the initial space "output_type", and
   converts them into the new space defined by the structure "space_info",
   which is then output as "coord_out".

   See Also: AFFINE3D_IMG.


.. py:function:: GoodVox2vol(img, dim)

   GOOD_VOX2VOL Transforms a VOX x TIME array into a volume, X x Y x Z x TIME, given a space described by "dim".

   imgvol = GOOD_VOX2VOL(img, dim) reshapes a VOX x TIME array "img" into
   an X x Y x Z x TIME array "imgvol", according to the dimensions of the
   space described by "dim".

   See Also: SPECTROSCOPY_IMG.


.. py:function:: rotate_cap(tpos_in, dTheta)

   ROTATE_CAP Rotates the cap in space.

   tpos_out = ROTATE_CAP(tpos_in, dTheta) rotates the cap grid given by
   "tpos_in" by the rotation vector "dTheta" (in degrees) and outputs it
   as "tpos_out".

   Dependencies: ROTATION_MATRIX.

   See Also: PLOTLRMESHES, SCALE_CAP.


.. py:function:: rotation_matrix(direction, theta)

   ROTATION_MATRIX Creates a rotation matrix.

   rot = ROTATION_MATRIX(direction, theta) generates a rotation matrix
   "rot" for the vector "direction" given an angle "theta" (in radians).

   See Also: ROTATE_CAP.


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


.. py:function:: rms_py(rms_input)

   RMS_PY Computes the RMS value for each column of a row vector.

   For matrices (N x M), rms_py(rms_input) is a row vector containing the RMS value from each column

   For complex input, rms_py separates the real and imaginary portions of each column of rms_input, squares them,
   and then takes the square root of the mean of that value

   For real input, the root mean square is calculated as follows:


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


.. py:function:: DynamicFilter(input_data, info_in, params, mode, save='no', pathToSave='./')

   This function is used to perform each step of the NeuroDOT PreProcessing Pipeline.

   The input "mode" is used to decide which step of the pipeline to perform, as well as 
   which figures to generate.

   Modes:
       :fft_lml: display Fourier transform of the logmean light levels
       :lml: display time traces of logmean light levels
       :fft_dt: display Fourier transform of detrended data
       :fft_dt_hp: display Fourier transform of detrended and high-pass filtered data
       :fft_superficial_signal: display Fourier transform of data which has undergone superficial signal regression
       :high_pass: display high-pass filtered data time traces
       :low_pass: display low-pass filtered data time traces
       :low_pass_fft: display Fourier transform of low-pass filtered data  
       :fft_lowpass_2: display Fourier transform of twice-low-pass filtered data  
       :superficial_signal: display time traces of data which has undergone superficial signal regression
       :fft_resample: display Fourier transform of resampled data
       :resample: display time traces of resampled data
       :ba: display time traces of block-averaged data
       :fft_ba: display Fourier transform of block-averaged data


.. py:function:: reconstruct_img(lmdata, iA)

   RECONSTRUCT_IMG Performs image reconstruction by wavelength using the inverted A-matrix.

   img = RECONSTRUCT_IMG(data, iA) takes the inverted VOX x MEAS
   sensitivity matrix "iA" and right-multiplies it by the logmeaned MEAS x
   TIME light-level matrix "lmdata" to reconstruct an image in voxel
   space. 

   The image is output in a VOX x TIME matrix "img".

   See Also: TIKHONOV_INVERT_AMAT, SMOOTH_AMAT, SPECTROSCOPY_IMG,
   FINDGOODMEAS.


.. py:function:: smooth_Amat(iA_in, dim, gsigma)

   SMOOTH_AMAT Performs Gaussian smoothing on a sensitivity matrix.

   iA_out = SMOOTH_AMAT(iA_in, dim, gsigma) takes the inverted VOX x MEAS
   sensitivity matrix "iA_in" and performs Gaussian smoothing in the 3D
   voxel space on each of the concatenated measurement matrices within,
   returning it as "iA_out". The user specifies the Gaussian filter
   half-width "gsigma".

   See Also: TIKHONOV_INVERT_AMAT, RECONSTRUCT_IMG, FINDGOODMEAS.


.. py:function:: spectroscopy_img(cortex_mu_a, E)

   SPECTROSCOPY_IMG Completes the Beer-Lambert law from a reconstructed image.

   img_out = SPECTROSCOPY_IMG(img_in, E) takes the reconstructed VOX x
   TIME x WL mua image "img_in" and multiplies it by the inverse
   of the extinction coefficient matrix "E" to create an output VOX x TIME
   x HB matrix "img_out", where HB 1 and 2 are the voxel-space time series
   images for HbO and HbR, respectively.

   See Also: RECONSTRUCT_IMG, AFFINE3D_IMG.


.. py:function:: Tikhonov_invert_Amat(A, lambda1, lambda2)

   TIKHONOV_INVERT_AMAT Inverts a sensitivity "A" matrix.

   iA = TIKHONOV_INVERT_AMAT(A, lambda1, lambda2) allows the user to
   specify the values of the "lambda1" and "lambda2" parameters in the
   inversion calculation. lambda2 for spatially-variant regularization,
   is optional.

   See Also: SMOOTH_AMAT, RECONSTRUCT_IMG, FINDGOODMEAS.


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


