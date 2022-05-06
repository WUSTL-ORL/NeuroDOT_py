#General imports
import numpy as np
import numpy.matlib as mlb
import scipy as scp


class sx4m:


    def affine3d_img(imgA, infoA, infoB, affine = np.eye(4), interp_type = 'nearest'):
        """
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
        """
        ## Parameters and Initialization.
        size_imgA = np.shape(imgA)
      
        if len(size_imgA) <= 3: 
            Nt = 1
        else:
            Nt = size_imgA[3]
        extrapval = 0

        ## Determine initial and target space coordinate ranges.
        # Initial space.
        centerA = infoA['center']
        nVxA = infoA['nVx']
        nVyA = infoA['nVy']
        nVzA = infoA['nVz']
        drA = infoA['mmppix']

        X = np.arange((-centerA[0] + nVxA * drA[0]), (-centerA[0]), -drA[0]) 
        Y = np.arange((-centerA[1] + nVyA * drA[1]), (-centerA[1]), -drA[1]) 
        Z = np.arange((-centerA[2] + nVzA * drA[2]), (-centerA[2]), -drA[2]) 

        # Target space.
        centerB = infoB['center']
        nVxB = infoB['nVx']
        nVyB = infoB['nVy']
        nVzB = infoB['nVz']
        drB = infoB['mmppix']

        x = np.arange((-centerB[0] + nVxB * drB[0]), (-centerB[0]), -drB[0]) 
        y = np.arange((-centerB[1] + nVyB * drB[1]), (-centerB[1]), -drB[1]) 
        z = np.arange((-centerB[2] + nVzB * drB[2]), (-centerB[2]), -drB[2])

        #make target space x, y, z column vectors (row vectors will throw an error in lines 205-207 because len(row_vector) = 1 as the first dimension of a row vector has a length of 1)
        x.shape = (x.shape[0], 1)
        y.shape = (y.shape[0], 1)
        z.shape = (z.shape[0], 1)


        ## Get all points in destination to sample
        [ygv, xgv, zgv] = np.meshgrid(y, x, z)
        xyz = np.squeeze(np.asarray([np.reshape(xgv, (np.size(xgv), 1), order = 'F').T, np.reshape(ygv, (np.size(ygv), 1), order = 'F').T, np.reshape(zgv, (np.size(zgv), 1), order = 'F').T])) #squeeze and asarray are used so that the row of 1's can be appended to xyz
        xyz = np.append(xyz, np.ones((1, np.size(xyz, 1))), axis = 0)

        ## Transform into source coordinates and remove extra column
        uvw = affine @ xyz
        uvw = np.transpose(uvw[0:3, :])

        ## Sample
        xi = np.reshape(uvw[:,0], (len(x), len(y), len(z)), order = 'F')
        yi = np.reshape(uvw[:,1], (len(x), len(y), len(z)), order = 'F')
        zi = np.reshape(uvw[:,2], (len(x), len(y), len(z)), order = 'F')


        ## Interpolate
        # scipy.interpolate.interpn requires initial space coordinates to all be strictly ascending, due to this we flip initial space corrdinates in the X dimension, as they are descending
        # interpn also requires the order of initial space inputs to match the dimension order of the input image, so the order is specified as X, Y , Z in python instead of Y, X, Z as it is in matlab
        # we also flip the x dimension of the input image to replicate what we're doing with the initial space X coordinates
        X = np.flip(X)
        imgA = np.flip(imgA, axis = 0)
        for k in range(0, Nt):
            imgB = scp.interpolate.interpn((X, Y, Z), np.squeeze(imgA[:, :, :]), np.array([xi, yi, zi]).T, method = interp_type, bounds_error = False, fill_value = extrapval)
        imgB1 = np.transpose(imgB)

        return imgB1


    def change_space_coords(coord_in, space_info, output_type = 'coord'):
        """
        CHANGE_SPACE_COORDS Applies a look up to change 3D coordinates into a new
        space.

        coord_out = CHANGE_SPACE_COORDS(coord_in, space_info, output_type) takes
        a set of coordinates "coord_in" of the initial space "output_type", and
        converts them into the new space defined by the structure "space_info",
        which is then output as "coord_out".

        See Also: AFFINE3D_IMG.
        """

        if len(np.shape(coord_in)) == 1:
            coord_in = np.reshape(coord_in,[1,-1])

        # Define the voxel space.
        nVxA = space_info['nVx']
        nVyA = space_info['nVy']
        nVzA = space_info['nVz']
        drA = space_info['mmppix']
        centerA = space_info['center'] # i.e., center = coordinate of center of voxel with index [-1,-1,-1]
        nV = np.shape(coord_in)[0]

        # Preallocate.
        coord_out = np.zeros(np.shape(coord_in))

        # Create coordinates for each voxel index.
        X = np.transpose(drA[0] * np.array(range(nVxA, 0, -1)) - centerA[0])
        Y = np.transpose(drA[1] * np.array(range(nVyA, 0, -1)) - centerA[1])
        Z = np.transpose(drA[2] * np.array(range(nVzA, 0, -1)) - centerA[2])

        # Convert coordinates to new space.
        if output_type == 'coord':
            #ATLAS/4DFP/ETC COORDINATE SPACE
            for j in range(0, nV):
                x = coord_in[j, 0]
                a = np.floor(x)
                if ((np.floor(x) >= 0) and (np.floor(x) <= nVxA)):
                    coord_out[j, 0] = X[int(np.floor(x))-1] - drA[0] * (x - np.floor(x))
                elif np.floor(x) < 0:
                    coord_out[j, 0] = X[0] - drA[0] * (x - 1)
                elif np.floor(x) > nVxA - 1:
                    coord_out[j, 0] = X[nVxA - 1] - drA[0] * (x - (nVxA - 1))

                y = coord_in[j, 1]
                if ((np.floor(y) >= 0) and (np.floor(y) <= nVyA)):
                    coord_out[j, 1] = Y[int(np.floor(y))-1] - drA[1] * (y - np.floor(y))
                elif np.floor(y) < 0:
                    coord_out[j, 1] = Y[0] - drA[1] * (y - 1)
                elif np.floor(y) > nVyA - 1:
                    coord_out[j, 1] = Y[nVyA - 1] - drA[1] * (y - (nVyA - 1))

                z = coord_in[j, 2]
                if ((np.floor(z) >= 0) and (np.floor(z) <= nVzA)):
                    coord_out[j, 2] = Z[int(np.floor(z))-1] - drA[2] * (z - np.floor(z))
                elif np.floor(z) < 0:
                    coord_out[j, 2] = Z[0] - drA[2] * (z - 1)
                elif np.floor(z) > nVzA - 1:
                    coord_out[j, 2] = Z[nVzA - 1] - drA[2] * (z - (nVzA - 1))
        elif output_type == 'idx': # MATLAB INDEX SPACE
            for j in range(0, nV):
                coord_out[j, 0] = np.argmin(abs(coord_in[j, 0] - X))
                coord_out[j, 1] = np.argmin(abs(coord_in[j, 1] - Y))
                coord_out[j, 2] = np.argmin(abs(coord_in[j, 2] - Z))

        elif output_type == 'idxC': # MATLAB INDEX SPACE WITH NO ROUNDING
            for j in range(0, nV):
                foo = np.argmin(abs(coord_in[j, 0] - X))
                coord_out[j, 0] = foo + (X[foo] - coord_in[j, 0]) / drA[0]
                foo = np.argmin(abs(coord_in[j, 1] - Y))
                coord_out[j, 1] = foo + (Y[foo] - coord_in[j, 1]) / drA[1]
                foo = np.argmin(abs(coord_in[j, 2] - Z))
                coord_out[j, 2] = foo + (Z[foo] - coord_in[j, 2]) / drA[2]

        return coord_out


    def GoodVox2vol(img, dim):
        """
        GOOD_VOX2VOL Turns a Good Voxels data stream into a volume.

        imgvol = GOOD_VOX2VOL(img, dim) reshapes a VOX x TIME array "img" into
        an X x Y x Z x TIME array "imgvol", according to the dimensions of the
        space described by "dim".

        See Also: SPECTROSCOPY_IMG.
        """
        ## Parameters and Initialization.
        Nvox = np.shape(img)[0]
        Nt = np.shape(img)[1]
        if np.logical_and(Nvox == 1, Nt > 1):
            img = img.T
            Nvox = np.shape(img)[0]
            Nt = np.shape(img)[1] 

        ## Stream image into good voxels.
        imgvol = np.zeros((dim['nVx'] * dim['nVy'] * dim['nVz'], Nt)) #something going wrong here
        imgvol[dim['Good_Vox']-1, :] = img

        ## Reshape image into the voxel space.
        imgvol = np.reshape(imgvol, (dim['nVx'], dim['nVy'], dim['nVz'], Nt), order = 'F')

        return imgvol


    def rotate_cap(tpos_in, dTheta):
        """
        ROTATE_CAP Rotates the cap in space.

        tpos_out = ROTATE_CAP(tpos_in, dTheta) rotates the cap grid given by
        "tpos_in" by the rotation vector "dTheta" (in degrees) and outputs it
        as "tpos_out".

        Dependencies: ROTATION_MATRIX.

        See Also: PLOTLRMESHES, SCALE_CAP.
        """
        ## Parameters and Initialization.
        centroid = np.mean(tpos_in, axis = 0) #this is causing the large error in output
        centroid_mat = mlb.repmat(centroid, tpos_in.shape[0], 1)
        Dx_axis = dTheta[0] # Pos rotates true right down
        Dy_axis = dTheta[1] # Pos rotates CCW from top
        Dz_axis = dTheta[2] # Pos rotates back of cap up
        d2r = np.pi / 180 #  Convert from degrees to radians


        ## Create rotation matrices.
        rotX = sx4m.rotation_matrix('x', Dx_axis * d2r)
        rotY = sx4m.rotation_matrix('y', Dy_axis * d2r)
        rotZ = sx4m.rotation_matrix('z', Dz_axis * d2r)

        ## Rotate around Centroid.
        rot = rotX @ rotY @ rotZ
        tpos_out = (tpos_in - centroid_mat) @ rot + centroid_mat

        return tpos_out


    def rotation_matrix(direction, theta):
        """
        ROTATION_MATRIX Creates a rotation matrix.
    
        rot = ROTATION_MATRIX(direction, theta) generates a rotation matrix
        "rot" for the vector "direction" given an angle "theta" (in radians).
        
        See Also: ROTATE_CAP.
        """
        ## Parameters and Initialization.
        rot = np.zeros(3)

        if direction == 'x':
            rot = np.array(([1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]), order = 'F')
        if direction == 'y':
            rot = np.array(([np.cos(theta), 0 , np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]), order = 'F')
        if direction == 'z':
            rot = np.array(([np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1],), order = 'F')
        
        return rot