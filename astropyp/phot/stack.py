import numpy as np

def reproject_image(img, x, y, new_x, new_y, tx_solution=None,
        dqmask=None, max_offset=3, subsampling=5, 
        show_plot=False, dqmask_min=0, bad_pix_val=1):
    """
    Reproject one image onto another using image coordinates
    
    Parameters
    ----------
    img: array-like
        Array containing the image to reproject
    x,y: float
        Coordinates at the center of the projection. If ``tx_solutions``
        is None, these must be in the original image coordinates. 
        Otherwise these must be in the reprojected image coordinates.
    new_x, new_y: array-like
        Coordinate positions for each pixel on the x and y axes in the
        image. If ``tx_solutions`` is None, these must be in the 
        original image coordinates. Otherwise these must be in the 
        reprojected image coordinates.
    tx_solution: `~astropyp.catalog.ImageSolution`, optional
        Astrometric solution to transform from the current image coordinates
        to the new image coordinates. If x, y, new_x, new_y are all in the
        original image coordinates, set ``tx_solution`` to None.
        *Default is None*
    dqmask: `~numpy.ndarray`, optional
        Data quality mask to apply to the reprojected image. This can
        either be in the coordinate system of ``img``, in which case
        ``reproject_dqmask=True`` or in the new coordinate system,
        in which case ``reproject_dqmask=False``.
    max_offset: int, optional
        Maximum number of pixels a the image is allowed to be moved to
        recenter on the maximum subpixel (in image pixels, not subpixels).
        *Default=3*
    subsampling: int
        Number of subdivisions of each pixel. *Default=5*
    show_plot: bool, optional
        Whether or not to show a plot of the reprojected image.
        *Default=False*
    dqmask_min: int, optional
        Minimum value of a data quality mask that is accepted.
        All pixels higher than ``dqmask_min`` will be masked in the
        reprojected image. *Default=0*
    bad_pix_val: int, optional
        Value to set bad pixels to in the dqmask. This is only 
        necessary if using a dqmask. *Default=1*
    
    Results
    -------
    modified_data: `~numpy.ndarray`
        Data in the reprojected coordinate system
    modified_dqmask: `~numpy.ndarray`
        dqmask in the reprojected coordinate system. This will be
        ``None`` if no dqmask was passed to the function.
    X0: `~numpy.ndarray`
        Coodinates on the X axis in the original coordinate system
        after it has been subsampled and re-centered
    Y0: `~numpy.ndarray`
        Coodinates on the Y axis in the original coordinate system
        after it has been subsampled and re-centered
    new_pos: tuple
        New position in the original coordinate system after it has
        been subsampled and re-centered
    """
    from scipy import interpolate
    from astropy.nddata.utils import extract_array
    import astropyp.utils
    src_shape = (len(new_y)/subsampling, len(new_x)/subsampling)
    
    # Convert the centroid position from destination coordinates
    # to current image coordinates and extract the subsampled image
    if tx_solution is not None:
        x0,y0 = tx_solution.transform_coords(x=x, y=y)
    else:
        x0,y0 = (x,y)
    data, X0, Y0, new_pos = astropyp.utils.misc.get_subpixel_patch(
        img, src_pos=(y0,x0), src_shape=src_shape, 
        max_offset=max_offset, subsampling=subsampling, normalize=False)
    if data is None:
        return None, dqmask, X0, Y0, new_pos
    Z = interpolate.RectBivariateSpline(Y0, X0, data)
    
    # Convert the coordinates from the destination to the
    # current image and extract the interpolated pixels
    if tx_solution is not None:
        X,Y = tx_solution.transform_coords(x=new_x, y=new_y)
    else:
        X,Y = (new_x, new_y)
    modified_data = Z(Y,X)
    modified_data = np.ma.array(modified_data)
    if dqmask is not None:
        modified_dqmask = reproject_dqmask(
            dqmask, new_pos[1], new_pos[0], X, Y, None,
            subsampling, show_plot, bad_pix_val)
        modified_data.mask = modified_dqmask>dqmask_min
    else:
        modified_dqmask = None
    if show_plot:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(modified_data, interpolation='none')
        plt.show()
    return modified_data, modified_dqmask, X0, Y0, new_pos

def reproject_dqmask(dqmask, x, y, new_x, new_y, tx_solution=None,
        subsampling=5, show_plot=False, bad_pix_val=1):
    """
    Reproject a data quality mask onto another using image coordinates
    
    Parameters
    ----------
    dqmask: array-like
        Array containing the image to reproject
    x,y: float
        Coordinates at the center of the projection. If ``tx_solutions``
        is None, these must be in the original image coordinates. 
        Otherwise these must be in the reprojected image coordinates.
    new_x, new_y: array-like
        Coordinate positions for each pixel on the x and y axes in the
        image. If ``tx_solutions`` is None, these must be in the 
        original image coordinates. Otherwise these must be in the 
        reprojected image coordinates.
    tx_solution: `~astropyp.catalog.ImageSolution`, optional
        Astrometric solution to transform from the current image coordinates
        to the new image coordinates. If x, y, new_x, new_y are all in the
        original image coordinates, set ``tx_solution`` to None.
        *Default is None*
    subsampling: int
        Number of subdivisions of each pixel. *Default=5*
    show_plot: bool, optional
        Whether or not to show a plot of the reprojected image.
        *Default=False*
    bad_pix_val: int, optional
        Value to set bad pixels to in the dqmask. This is only 
        necessary if using a dqmask. *Default=1*
    
    Results
    -------
    modified_data: `~numpy.ndarray`
        dqmask in the reprojected coordinate system. This will be
        ``None`` if no dqmask was passed to the function.
    """
    from scipy import interpolate
    from astropy.nddata.utils import extract_array
    from scipy.ndimage import zoom
    
    src_shape = (len(new_y)/subsampling, len(new_x)/subsampling)
    # Convert the centroid position from destination coordinates
    # to current image coordinates and extract the subsampled image
    if tx_solution is not None:
        x0,y0 = tx_solution.transform_coords(x=x, y=y)
    else:
        x0,y0 = (x,y)
    dqdata = extract_array(dqmask, src_shape, (y0,x0))
    # Set any edge values to the bad_pix_val and scale the dqmask
    # with a nearest neighbor (pixelated) interpolation
    dqdata[dqdata<0] = bad_pix_val
    dqdata = zoom(dqdata, subsampling, order=0)
    # Extract the projected patch
    radius = 0.5*(np.array(src_shape)-1)
    X = np.linspace(x0-radius[1], x0+radius[1], dqdata.shape[1])
    Y = np.linspace(y0-radius[0], y0+radius[0], dqdata.shape[0])
    X,Y = np.meshgrid(X,Y)
    coords = zip(Y.flatten(),X.flatten())
    
    # Convert the coordinates from the destination to the
    # current image and extract the interpolated pixels
    if tx_solution is not None:
        X1,Y1 = tx_solution.transform_coords(x=new_x, y=new_y)
    else:
        X1,Y1 = (new_x, new_y)
    X1,Y1 = np.meshgrid(X1,Y1)
    new_coords = zip(Y1.flatten(),X1.flatten())
    modified_data = interpolate.griddata(coords, 
        dqdata.flatten(), 
        new_coords, 
        method='nearest').reshape(dqdata.shape)
    if show_plot:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(modified_data, interpolation='none')
        plt.show()
    return modified_data

def stack_source(images, x, y, src_shape, index, tx_solutions,
        dqmasks=None, max_offset=3, subsampling=5, combine_method='mean', 
        show_plots=False, dqmask_min=0, bad_pix_val=1):
    """
    Stack all the images of a given source.
    
    Parameters
    ----------
    images: list of `~numpy.ndarray`'s
        Images to stack
    x,y: float
        Central pixel of reprojected windows
    src_shape: tuple of integers
        Shape (y,x) of the patch to extract from each image. Typically
        this is 2*aper_radius+1.
    index: int
        Index in ``images`` and ``dqmasks`` of the image the others
        are projected onto for the stack.
    tx_solutions: list of `~astropyp.catalog.ImageSolution`s
        Transformations to convert each image to the projected coordinates
    dqmasks: list of `~numpy.ndarrays`'s, optional
        Dqmasks for images to stack
    max_offset: int, optional
        Maximum number of pixels a the image is allowed to be moved to
        recenter on the maximum subpixel (in image pixels, not subpixels).
        *Default=3*
    subsampling: int
        Number of subdivisions of each pixel. *Default=5*
    combine_method: string
        Method to use for the stack. This can be either 'median' or 'mean'.
        *Default is 'mean'*
    show_plots: bool, optional
        Whether or not to show a plots of the reprojected images and
        dqmasks. *Default=False*
    dqmask_min: int, optional
        Minimum value of a data quality mask that is accepted.
        All pixels higher than ``dqmask_min`` will be masked in the
        reprojected image. *Default=0*
    bad_pix_val: int, optional
        Value to set bad pixels to in the dqmask. This is only 
        necessary if using a dqmask. *Default=1*
    """
    from astropyp.catalog import Catalog
    from scipy.ndimage import zoom
    from astropy.nddata.utils import extract_array
    import astropyp.utils
    
    if combine_method=='median':
        combine = np.ma.median
    elif combine_method=='mean':
        combine = np.ma.mean
    else:
        raise Exception(
            "Combine method must be either 'median' or 'mean'")
    
    # Get the patch from the original image
    data, x_new, y_new, new_pos = astropyp.utils.misc.get_subpixel_patch(
        images[index], src_pos=(y,x), src_shape=src_shape, 
        max_offset=max_offset, subsampling=subsampling, normalize=False)
    
    if dqmasks is not None:
        # Get the data quality mask for the original image and
        # set any edge values to the bad_pix_val, and scale the dqmask
        # with a nearest neighbor (pixelated) interpolation
        dqmask = extract_array(dqmasks[index], src_shape, (y,x))
        dqmask[dqmask<0] = bad_pix_val
        dqmask = zoom(dqmask, subsampling, order=0)
    
    if data is not None:
        data = np.ma.array(data)
        if dqmasks is not None:
            data.mask = dqmask>dqmask_min
    
    if show_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        if dqmasks is not None:
            plt.imshow(dqmask, interpolation='none')
            plt.show()
        if data is not None:
            plt.imshow(data, interpolation='none')
            plt.show()
    # Reproject the images
    modified_data = [None, data, None]
    modified_dqmask = [None, dqmask, None]
    for n in [m for m in range(len(images)) if m!=index]:
        modified_data[n],modified_dqmask[n], X0, Y0, new_pos = reproject_image(
            images[n], x, y, x_new, y_new, tx_solutions[n], dqmasks[n],
            max_offset, subsampling, show_plots, dqmask_min, bad_pix_val)
    
    # Only stack non-null images
    modified_data = [m for m in modified_data if m is not None]
    if len(modified_data)==0:
        return None, None
    elif len(modified_data)==1:
        return modified_data[0], modified_dqmask[0]
    
    # Stack the images and combine the data quality masks
    stack = np.ma.array(modified_data)
    stack = combine(stack, axis=0)
    dqmask = modified_data[0].mask
    for n in range(1,len(modified_data)):
        dqmask = np.bitwise_and(dqmask, modified_data[n].mask)
    
    if show_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        Xmesh, Ymesh = np.meshgrid(x_new, y_new)
        fig = plt.figure(figsize=(6, 6))
        ax=fig.add_subplot(1,1,1, projection='3d')
        ax.plot_wireframe(Xmesh, Ymesh, stack)
        plt.show()
        plt.contour(Xmesh,Ymesh,stack)
        plt.axis('equal')
        plt.show()
        plt.imshow(stack, interpolation='none')
        plt.show()
        plt.imshow(dqmask, interpolation='none')
        plt.show()
    
    return stack, dqmask