:py:mod:`neuro_dot.Spatial_Transforms`
======================================

.. py:module:: neuro_dot.Spatial_Transforms


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   neuro_dot.Spatial_Transforms.affine3d_img
   neuro_dot.Spatial_Transforms.change_space_coords
   neuro_dot.Spatial_Transforms.GoodVox2vol
   neuro_dot.Spatial_Transforms.rotate_cap
   neuro_dot.Spatial_Transforms.rotation_matrix



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


