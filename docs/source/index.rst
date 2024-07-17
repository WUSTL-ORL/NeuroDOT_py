.. NeuroDOT_py documentation master file, created by
   sphinx-quickstart on Wed May  8 11:24:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to NeuroDOT_Py's Documentation
=======================================
NeuroDOT is an extensible toolbox for efficient brain mapping using diffuse optical tomography (DOT), which is
based on functional near-infrared spectroscopy (fNIRS).
NeuroDOT is compatible with a wide variety of fNIRS and DOT systems, ranging from very sparse to high density arrays.
This documentation provides tutorials for using the NeuroDOT modules in your own Python code, as well as an searchable API for fast reference.

.. image:: docs/images/overview.png
   :width: 1200px
   :height: 800px
   :scale: 50 %
   :alt: alternate text
   :align: center

.. toctree::
   :hidden:
   :maxdepth: 8
   :caption: Tutorials

   tutorials
   E:/Emma/neuroDOT/NeuroDOT_py/docs/source/getting_started.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Data Types
   
   data_types

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Development

   development

.. toctree::
   :hidden:
   :maxdepth: 5
   :caption: Functions:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. autosummary::
   :toctree: _autosummary
   './build/html/_sources/_autosummary/neuro_dot.Visualizations.rst'
   './build/html/_sources/_autosummary/neuro_dot.Temporal_Transforms.rst'
   './build/html/_autosummary/neuro_dot.Spatial_Transforms.rst'
   './build/html/_autosummary/neuro_dot.Reconstruction.rst'
   './build/html/_autosummary/neuro_dot.Matlab_Equivalent_Functions.rst'
   './build/html/_autosummary/neuro_dot.Light_Modeling.rst'
   './build/html/_autosummary/neuro_dot.File_IO.rst'
   './build/html/_autosummary/neuro_dot.DynamicFilter.rst'
   './build/html/_autosummary/neuro_dot.Analysis.rst'

