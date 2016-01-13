from __future__ import division
from collections import OrderedDict
import numpy as np
import logging
import warnings

from astropy.extern import six
from astropy import table

logger = logging.getLogger('astropyp.utils.misc')

def isnumber(x):
    try:
        float(x)
    except (ValueError, TypeError) as e:
        return False
    return True

uint8_flags = OrderedDict([
    (128, 'Bit 8'),
    (64, 'Bit 7'),
    (32, 'Bit 6'),
    (16, 'Bit 5'),
    (8, 'Bit 4'),
    (4, 'Bit 3'),
    (2, 'Bit 2'),
    (1, 'Bit 1')
])

class InternalFlags:
    """
    InternalFlags must be initialized with an OrderedDict, such
    as uint8_flags.  
    """
    def __init__(self, flags=uint8_flags):
        self.flags = flags
        if isinstance(flags, six.string_types):
            self.set_flags(flags)
        self.flag_type = self.set_flag_type()
    def set_flag_type(self):
        bit_keys = [1,2,4,8,16,32,64,128]
        if np.all([k in bit_keys for k in self.flags.keys()]):
            self.flag_type = 'bits'
        else:
            self.flag_type = 'values'
        return self.flag_type
    def set_flags(self, flag_module):
        if flag_module=='sex':
            import astropyp.wrappers.astromatic.utils as utils
            self.flags = utils.sex_internal_flags
        elif flag_module=='decam early':
            import astropyp.instruments.decam.utils as utils
            self.flags = utils.decam_flags_early
        elif flag_module=='decam':
            import astropyp.instruments.decam.utils as utils
            self.flags = utils.decam_flags
        else:
            raise Exception("Flag {0} has not yet been added".format())
        self.set_flag_type()
    def get_flag_info(self):
        if self.flag_type=='bits':
            key = 'Bits'
        else:
            key = 'Value'
        result = table.Table(
            [self.flags.keys(), [v for k,v in self.flags.items()]], 
            names=(key, 'Description'))
        return result
    def get_flags(self, flags):
        if self.flag_type=='bits':
            binflags = flags.astype(np.uint8)
            binflags = binflags.reshape((binflags.shape[0],1))
            binflags = np.unpackbits(binflags, axis=1)
            tbl = table.Table()
            for n,f in enumerate(self.flags):
                tbl[str(f)] = np.array(binflags[:,n], dtype=bool)
        else:
            tbl = table.Table()
            for n,f in enumerate(self.flags):
                tbl[str(f)] = flags==f
        return tbl

def update_ma_idx(arr, idx):
    new_array = np.ma.array(arr)[idx]
    new_array.mask = new_array.mask | idx.mask
    return new_array

from astropy.nddata import overlap_slices
def extract_array(array_large, shape, position, mode='partial',
                  fill_value=np.nan, return_position=False,
                  subsampling=None, order=3, smoothing=0,
                  masking=False, return_slices=False):
    """
    Extract a smaller array of the given shape and position from a
    larger array.

    Parameters
    ----------
    array_large : `~numpy.ndarray`
        The array from which to extract the small array.
    shape : tuple or int
        The shape of the extracted array (for 1D arrays, this can be an
        `int`).  See the ``mode`` keyword for additional details.
    position : tuple of numbers or number
        The position of the small array's center with respect to the
        large array.  The pixel coordinates should be in the same order
        as the array shape.  Integer positions are at the pixel centers
        (for 1D arrays, this can be a number).
    mode : {'partial', 'trim', 'strict'}, optional
        The mode used for extracting the small array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the
        small array and the large array is sufficient.  For the
        ``'strict'`` mode, the small array has to be fully contained
        within the large array, otherwise an
        `~astropy.nddata.utils.PartialOverlapError` is raised.   In all
        modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the small array that do not overlap with the large
        array will be filled with ``fill_value``.  In ``'trim'`` mode
        only the overlapping elements are returned, thus the resulting
        small array may be smaller than the requested ``shape``.
    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the extracted
        small array that do not overlap with the input ``array_large``.
        ``fill_value`` must have the same ``dtype`` as the
        ``array_large`` array.
    return_position : boolean, optional
        If `True`, return the coordinates of ``position`` in the
        coordinate system of the returned array.

    Returns
    -------
    array_small : `~numpy.ndarray`
        The extracted array.
    new_position : tuple
        If ``return_position`` is true, this tuple will contain the
        coordinates of the input ``position`` in the coordinate system
        of ``array_small``. Note that for partially overlapping arrays,
        ``new_position`` might actually be outside of the
        ``array_small``; ``array_small[new_position]`` might give wrong
        results if any element in ``new_position`` is negative.

    Examples
    --------
    We consider a large array with the shape 11x10, from which we extract
    a small array of shape 3x5:

    >>> import numpy as np
    >>> from astropy.nddata.utils import extract_array
    >>> large_array = np.arange(110).reshape((11, 10))
    >>> extract_array(large_array, (3, 5), (7, 7))
    array([[65, 66, 67, 68, 69],
           [75, 76, 77, 78, 79],
           [85, 86, 87, 88, 89]])
    """

    if np.isscalar(shape):
        shape = (shape, )
    if np.isscalar(position):
        position = (position, )

    if mode not in ['partial', 'trim', 'strict']:
        raise ValueError("Valid modes are 'partial', 'trim', and 'strict'.")
    large_slices, small_slices = overlap_slices(array_large.shape,
                                                shape, position, mode=mode)
    extracted_array = array_large[large_slices]
    if return_position:
        new_position = [i - s.start for i, s in zip(position, large_slices)]
    if subsampling is None:
        # Extracting on the edges is presumably a rare case, so treat special here
        if (extracted_array.shape != shape) and (mode == 'partial'):
            extracted_array = np.zeros(shape, dtype=array_large.dtype)
            extracted_array[:] = fill_value
            extracted_array[small_slices] = array_large[large_slices]
            if return_position:
                new_position = [i + s.start for i, s in zip(new_position,
                                                            small_slices)]
    else:
        # Subsample the array, centered on the position
        try:
            from scipy import interpolate
        except ImportError:
            raise ImportError(
                "You must have scipy installed to use the subpixel method")
        src_pos = np.array(position)
        pix_pos = np.round(src_pos)
        x_radius = .5*(extracted_array.shape[1]-1)
        y_radius = .5*(extracted_array.shape[0]-1)
        X = np.linspace(pix_pos[1]-x_radius, pix_pos[1]+x_radius, 
            extracted_array.shape[1])
        Y = np.linspace(pix_pos[0]-y_radius, pix_pos[0]+y_radius, 
            extracted_array.shape[0])
        # Create an interpolating function to scale the data
        data_func = interpolate.RectBivariateSpline(Y, X, extracted_array, 
            kx=order, ky=order, s=smoothing)
        
        # In the case of a partial overlap, center the extracted array on an 
        # array of fill_value pixels
        if (extracted_array.shape != shape) and (mode == 'partial'):
            sub_shape = (shape[0]*subsampling, shape[1]*subsampling)
            large_shape = (array_large.shape[0]*subsampling, 
                           array_large.shape[1]*subsampling)
            large_slices, small_slices = overlap_slices(large_shape, sub_shape, 
                np.array(position)*subsampling, mode='partial')
            sub_array = np.zeros(sub_shape, dtype=array_large.dtype)
            sub_array[:] = fill_value
            
            x_width = small_slices[1].stop-small_slices[1].start
            y_width = small_slices[0].stop-small_slices[0].start
            sub_x_radius = 0.5*(x_width)/subsampling
            sub_y_radius = 0.5*(y_width)/subsampling
            X = np.linspace(src_pos[1]-sub_x_radius, 
                            src_pos[1]+sub_x_radius, 
                            x_width)
            Y = np.linspace(src_pos[0]-sub_y_radius, 
                            src_pos[0]+sub_y_radius, 
                            y_width)
            sub_array[small_slices] = data_func(Y,X)
            extracted_array = sub_array
        else:
            # Get coordinates for the extracted subsampled array centered 
            #on the requested position
            X = np.linspace(src_pos[1]-x_radius, src_pos[1]+x_radius, 
                extracted_array.shape[1]*subsampling)
            Y = np.linspace(src_pos[0]-y_radius, src_pos[0]+y_radius, 
                extracted_array.shape[0]*subsampling)
            extracted_array = data_func(Y,X)
            large_slices = (slice(0,extracted_array.shape[0]),
                           slice(0,extracted_array.shape[1]))
            small_slices = (slice(0,extracted_array.shape[0]),
                           slice(0,extracted_array.shape[1]))
    if masking and (np.any(extracted_array==fill_value) or
            np.any(np.isnan(extracted_array))):
        extracted_array = np.ma.array(extracted_array)
        extracted_array.mask = extracted_array==fill_value
    result = [extracted_array]
    if return_position:
        result.append(tuple(new_position))
    if return_slices:
        result.append((large_slices, small_slices))
    if len(result)==1:
        result = result[0]
    return result

def get_subpixel_patch(img_data, src_pos=None, src_shape=None, 
        max_offset=3, subsampling=5, normalize=False, xmin=None, 
        ymin=None, xmax=None, ymax=None, order=5,
        window_sampling=100, window_radius=1, smoothing=0,
        show_plots=False):
    """
    Interpolate an image by subdividing each pixel into ``subpixels``
    pixels. It also (optionally) centers the patch on the pixel with 
    the maximum flux.
    
    Parameters
    ----------
    img_data: array-like
        The image data to use for extraction
    src_pos: tuple, optional
        Position (y,x) of the center of the patch. Either obj_pos must
        be given or one of the following: xmin,ymin and obj_shape;
        xmax,ymax and obj_shape; or xmin,xmax,ymin,ymax.
    src_shape: tuple, optional
        Shape (y size, x size) of the patch in the *original* image.
        Each of these must be an odd number
    max_offset: float, optional
        Size of the border (in pixels from the original image) to extract
        from the patch of the original image. This allows the function
        to re-center the patch on the new maximum, which may be slightly
        different. If ``offset_buffer=0`` or the change in position is
        greater than the ``offset_buffer``, then re-centering is cancelled.
        This is necessary because in crowded fields multiple objects
        may be overlapping and this prevents fainter sources from being
        re-centered on their brighter neighbors. For this reason it is
        good to set ``offset_buffer`` small so that only minor corrections
        will be made.
        *Default=2*
    subpixels: integer, optional
        Number of pixels to subdivide the original image. This
        must be an odd number. *Default=5*
    normalize: bool, optional
        Whether or not to normalize the data so that the maximum
        value is 1. *Default=True*
    xmin,ymin: tuple, optional
        Location of the top left corner of the patch to extract. If these
        arguments are used then either xmax,ymax or obj_shape must also
        be set.
    xmax, ymax: tuple, optional
        Location of the top left corner of the patch to extract. If these
        arguments are used then either xmin,ymin or obj_shape must also
        be set.
    order: tuple, optional
        Order of the polynomial to fit to the data. *Default=5*
    window_radius: int, optional
        Radius of the window (in image pixels) to use to center the patch.
        *Default=1*
    window_sampling: int, optional
        How much to subdivide the window used to recenter the
        source. *Default=100*
    show_plots: bool, optional
        Whether or not to show a plot of the array centered on the
        sources position. *Default=False*
    Returns
    -------
    obj_data: `numpy.ndarray`
        The subdivided patch of the image centered on the maximum value.
    X,Y: `~numpy.ndarray`
        x and y coordinates of the patch
    """
    from astropy.nddata.utils import extract_array
    try:
        from scipy import interpolate
    except ImportError:
        raise Exception(
            "You must have scipy installed to use the subpixel method")
    
    err_msg = "To extract a patch one of the following combinations " \
        "must be given: obj_pos and obj_shape; xmin,ymin and obj_shape;" \
        "xmax,ymax and obj_shape; or xmin,xmax,ymin,ymax,"
        
    # Allow the user to choose the patch based on its center position
    # and shape, bounds, or a single x,y bound and shape
    if src_pos is None:
        if xmin is not None and ymin is not None:
            if xmax is not None and ymax is not None:
                src_pos = (.5*(ymin+ymax),.5*(xmin+xmax))
                src_shape = (ymax-ymin, xmax-xmin)
            elif src_shape is not None:
                src_pos = (ymin+.5*src_shape[0], xmin+.5*src_shape[1])
            else:
                raise Exception(err_msg)
        elif xmax is not None and ymax is not None:
            if src_shape is not None:
                src_pos = (ymax-.5*src_shape[0], xmax-.5*src_shape[1])
            else:
                raise Exception(err_msg)
    elif src_shape is None:
        raise Exception(err_msg)
    obj_pos = np.array(src_pos)
    obj_shape = np.array(src_shape)
    
    # use a pixelated position (since this corresponds to the indices of the 
    # array) to extract that data, which will be used for interpolation
    pix_pos = np.round(obj_pos)
    
    # Extract object data from the image with a buffer in case the maximum is 
    # not in the center and create an interpolation function
    data = extract_array(img_data, tuple(obj_shape+2*max_offset), pix_pos)
    x_radius = .5*(data.shape[1]-1)
    y_radius = .5*(data.shape[0]-1)
    X = np.linspace(pix_pos[1]-x_radius, pix_pos[1]+x_radius, 
        data.shape[1])
    Y = np.linspace(pix_pos[0]-y_radius, pix_pos[0]+y_radius, 
        data.shape[0])
    data_func = interpolate.RectBivariateSpline(Y, X, data, 
        kx=order, ky=order, s=smoothing)
    
    # If the extracted array contains NaN values return a masked array
    if not np.all(np.isfinite(data)) or max_offset==0:
        x_radius = .5*(obj_shape[1]-1)
        y_radius = .5*(obj_shape[0]-1)
        X = np.linspace(obj_pos[1]-x_radius, obj_pos[1]+x_radius, 
            obj_shape[1]*subsampling)
        Y = np.linspace(obj_pos[0]-y_radius, obj_pos[0]+y_radius, 
            obj_shape[0]*subsampling)
        new_pos = src_pos
        
        if max_offset==0:
            # Get the interpolated information centered on the source position
            obj_data = data_func(Y, X)
        else:
            w = "The patch for the source at {0} ".format(obj_pos)
            w += "contains NaN values, cannot interpolate"
            warnings.warn(w)
            obj_data = None
    else:
        dx = max_offset*subsampling
        X = np.linspace(obj_pos[1]-dx, obj_pos[1]+dx, 
            (2*dx+1))
        Y = np.linspace(obj_pos[0]-dx, obj_pos[0]+dx, 
            (2*dx+1))
        Z = data_func(Y, X)
    
        # Calculate the number of subpixels to move the center
        peak_idx = np.array(np.unravel_index(np.argmax(Z), Z.shape))
        center = .5*(np.array(Z.shape)-1)
        dpeak = np.abs(center-peak_idx)

        # Get the the interpolated image centered on the maximum pixel value
        center_pos = obj_pos-dpeak/subsampling
        X = np.linspace(center_pos[1]-max_offset, 
            center_pos[1]+max_offset, window_sampling)
        Y = np.linspace(center_pos[0]-max_offset, 
            center_pos[0]+max_offset, window_sampling)
        Z = data_func(Y,X)
        if show_plots:
            import matplotlib.pyplot as plt
            plt.imshow(Z, interpolation='none')
            plt.title("before centering")
            plt.show()
        yidx,xidx = np.unravel_index(np.argmax(Z), Z.shape)
        new_pos = (Y[yidx], X[xidx])
        
        # Extract the array centered on the new position
        x_radius = .5*(obj_shape[1]-1)
        y_radius = .5*(obj_shape[0]-1)
        X = np.linspace(new_pos[1]-x_radius, new_pos[1]+x_radius, 
            obj_shape[1]*subsampling)
        Y = np.linspace(new_pos[0]-y_radius, new_pos[0]+y_radius, 
            obj_shape[0]*subsampling)
        obj_data = data_func(Y, X)
        if show_plots:
            import matplotlib.pyplot as plt
            plt.imshow(obj_data, interpolation='none')
            plt.title("after centering")
            plt.show()
    
        peak_idx = np.unravel_index(np.argmax(obj_data), obj_data.shape)
        center = (int((obj_data.shape[0]-1)/2), int((obj_data.shape[1]-1)/2.))
        dpeak = ((center[0]-peak_idx[0])/subsampling, 
            (center[1]-peak_idx[1])/subsampling)
    
    # Normalize the data so that each source will be on the same scale
    if normalize and obj_data is not None:
        obj_data = obj_data/np.max(obj_data)
    return obj_data, X, Y, new_pos