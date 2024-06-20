:py:mod:`neuro_dot.Visualizations`
==================================

.. py:module:: neuro_dot.Visualizations


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Visualizations.adjust_brain_pos
   neuro_dot.Visualizations.applycmap
   neuro_dot.Visualizations.DrawColoredSynchPoints
   neuro_dot.Visualizations.nlrGrayPlots_220324
   neuro_dot.Visualizations.PlotInterpSurfMesh
   neuro_dot.Visualizations.Plot_RawData_Cap_DQC
   neuro_dot.Visualizations.defineticks
   neuro_dot.Visualizations.PlotFalloffData
   neuro_dot.Visualizations.PlotFalloffLL
   neuro_dot.Visualizations.PlotLRMeshes
   neuro_dot.Visualizations.Plot_RawData_Metrics_I_DQC
   neuro_dot.Visualizations.Plot_RawData_Metrics_II_DQC
   neuro_dot.Visualizations.Plot_RawData_Time_Traces_Overview
   neuro_dot.Visualizations.cylinder
   neuro_dot.Visualizations.PlotCapData
   neuro_dot.Visualizations.PlotCapGoodMeas
   neuro_dot.Visualizations.PlotCapMeanLL
   neuro_dot.Visualizations.PlotCapPhysiologyPower
   neuro_dot.Visualizations.PlotSlices_correct_orientation
   neuro_dot.Visualizations.PlotSlices
   neuro_dot.Visualizations.PlotSlicesTimeTrace
   neuro_dot.Visualizations.PlotTimeTraceData
   neuro_dot.Visualizations.vol2surf_mesh



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


