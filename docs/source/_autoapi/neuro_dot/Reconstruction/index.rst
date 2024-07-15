:py:mod:`neuro_dot.Reconstruction`
==================================

.. py:module:: neuro_dot.Reconstruction


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Reconstruction.reconstruct_img
   neuro_dot.Reconstruction.smooth_Amat
   neuro_dot.Reconstruction.spectroscopy_img
   neuro_dot.Reconstruction.Tikhonov_invert_Amat



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


