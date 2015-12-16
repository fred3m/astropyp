import numpy as np
import logging

from astropy.extern import six
from astropy.nddata.utils import extract_array, subpixel_indices
from astropy.coordinates import SkyCoord
import astropy.units as apu
from astropy import table
from astropy.modeling import Fittable2DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling.fitting import LevMarLSQFitter

logger = logging.getLogger('astropyp.phot.psf')

def get_subpixel_patch(img_data, obj_pos, obj_shape, offset_buffer=3, 
        subpixels=5, normalize=True):
    """
    Interpolate an image by subdividing each pixel into ``subpixels``
    pixels. It also (optionally) centers the patch on the pixel with 
    the maximum flux.
    
    Parameters
    ----------
    img_data: array-like
        The image data to use for extraction
    obj_pos: tuple
        Position (y,x) of the center of the patch
    obj_shape: tuple
        Shape (y size, x size) of the patch in the *original* image.
        Each of these must be an odd number
    offset_bufffer: integer
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
    subpixels: integer
        Number of pixels to subdivide the original image. This
        must be an odd number. *Default=5*
    normalize: bool
        Whether or not to normalize the data so that the maximum
        value is 1.
    
    Returns
    -------
    obj_data: `numpy.ndarray`
        The subdivided patch of the image centered on the maximum value.
    """
    try:
        from scipy import interpolate
    except ImportError:
        raise Exception(
            "You must have scipy installed to use the subpixel method")
    # Extract object data from the image with a buffer in case the maximum is 
    # not in the center
    data = extract_array(img_data, 
        (obj_shape[0]+2*offset_buffer, obj_shape[1]+2*offset_buffer), 
        obj_pos)
    X = np.arange(0, data.shape[1]*subpixels, subpixels)
    Y = np.arange(0, data.shape[0]*subpixels, subpixels)
    data_func = interpolate.RectBivariateSpline(X, Y, data)
    X = np.arange(0, data.shape[1]*subpixels, 1)
    Y = np.arange(0, data.shape[0]*subpixels, 1)
    Z = data_func(X, Y, data)
    
    # Get the indices of the maximum value, This may be slightly different from
    # the center so we will likely have to re-align the image
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    center = (int((Z.shape[0]-1)/2), int((Z.shape[1]-1)/2.))
    dpeak = ((center[0]-peak_idx[0])/subpixels, 
        (center[1]-peak_idx[1])/subpixels)
    
    # Get the the interpolated image centered on the maximum data value
    # (if the maximum is within the offset buffer)
    if ((dpeak[0]<offset_buffer) & 
            (dpeak[1]<offset_buffer)):
        obj_data = extract_array(Z, (obj_shape[0]*subpixels, 
            obj_shape[1]*subpixels), peak_idx)
        peak_idx = np.unravel_index(np.argmax(obj_data), obj_data.shape)
    else:
        logger.warn(
            'Unable to re-center, maximum is at'
            ' {0} but offset buffer is {1}'.format(
            dpeak, offset_buffer))
        obj_data = Z
    
    # Normalize the data so that each source will be on the same scale
    if normalize:
        obj_data = obj_data/np.max(obj_data)
    return obj_data

def select_psf_sources(img_data, sources, aper_radius=None,
        min_flux=None, min_amplitude=None, min_dist=None, max_ratio=None, 
        edge_dist=None, verbose=True, ra='ra', dec='dec', a='a', b='b',
        x='x', y='y', radius='aper_radius', flux='aper_flux', peak='peak',
        units='deg', badpix_flags=[], max_flag=0):
    """
    From an image and a list of sources, generate a list of high SNR
    sources with circular profiles that are likely to be good sources
    to generate the psf.
    
    Parameters
    ----------
    img_data: array-like
        The image data
    sources: `~astropy.table.Table`
        A list of sources
    aper_radius: integer, optional
        Radius to use to calculate psf flux. If no ``aperture_radius``
        is given this defaults to the maximum radius in the the
        list of sources
    min_flux: integer, optional
        Minimum flux to be considered an valid source. If no
        ``min_flux`` is given then no cuts will be made based on
        the total flux
    min_aplitude: integer, optional
        Minimum peak flux of a pixel in for a psf source. If no 
        ``min_amplitude`` is given then no cuts will be made based
        on maximum flux in a pixel.
    min_dist: integer, optional
        Minimum distance (in pixels) to the nearest neighbor
        for a PSF source. If no ``min_dist`` is specified the
        function will set ``min_dist=3*aperture_radius``.
    max_ratio: float
        Maximum ratio of a sources two axes to be considered 
        sufficiently elliptical. To calculate max_ratio,
        ``a`` and ``b`` must be specified. The default value
        is the median(a/b)+stddev(a/b) for all of the sources
    edge_dist: integer, optional
        Maximum distance (in pixels) from the center of a source to the
        edge of the image. This defaults to the ``aperture_radius``+1.
    verbose: bool, optional
        Whether or not to log the results of the various cuts.
        *Default value is True*
    ra,dec: string or array-like, optional
        Arrays of coordinates or names of coordinate columns in ``sources``.
        *Default is ``ra='ra'``, ``dec='dec'``*
    x,y: string or array-like, optional
        Arrays of coordinates or names of coordinate columns in ``sources``.
        *Default is ``x='x'``, ``y='y'``*
    a,b: string or array-like, optional
        Arrays or names of columns in ``sources`` for the axes widths of
        the psf sources (from SEP extract).
        *Default ``a='a'``, ``b='b'``*
    radius: string or array-like, optional
        Array or name of the aperture radius column in ``sources``.
        *Default ``radius='aper_radius'``*
    flux: string or array-like, optional
        Array of name of the aperture flux column in ``sources``.
        *Default ``flux='aper_flux'``*
    peak: string or array-like, optional
        Array or name of the peak column in ``sources``.
        *Default ``peak='peak'``*
    units: string or `~astropy.units.unit`
        Units of the ra and dec columns. *Default value is 'deg'*
    badpix_flags: list of strings or list of array-like, optional
        A list of column names in ``sources`` or a list of arrays
        with the badpixel flags for each source.
        *Default is an empty list*
    max_flag: integer, optional
        Maximum value of a bad pixel flag to be considered good.
        *Default value is 0*
    
    Returns
    -------
    psf_sources: `~astropy.table.Table`
        Table of psf sources
    flags: `~astropy.table.Table`
        Table of flags for all of the sources
    """
    # If the user passed column names instead of arrays, get the
    # appropriate columns from ``sources``
    columns = [x,y,a,b,ra,dec, flux, peak]
    if isinstance(x, six.string_types):
        x = sources[x]
    if isinstance(y, six.string_types):
        y = sources[y]
    if isinstance(a, six.string_types):
        a = sources[a]
    if isinstance(b, six.string_types):
        b = sources[b]
    if isinstance(ra, six.string_types):
        ra = sources[ra]
    if isinstance(dec, six.string_types):
        dec = sources[dec]
    if isinstance(flux, six.string_types):
        flux = sources[flux]
    if isinstance(peak, six.string_types):
        peak = sources[peak]
            
    # Get the total number of sources
    if hasattr(x, 'shape'):
        rows = x.shape[0]
    else:
        rows = len(x)
    
    # If the user didn't specify the aperture radius, 
    # use the maximum aperture radius
    # (note, to build the prf properly it is important that all of the sources 
    # use the same aperture radius)
    if aper_radius is None:
        if isinstance(radius, six.string_types):
            radius = sources[radius]
        aper_radius = np.max(radius)
    
    # Only select high signal to noise sources
    min_flux_flag = ~np.isnan(flux)
    if min_flux is not None:
        min_flux_flag = min_flux_flag & (flux>min_flux)
    if min_amplitude is not None:
        min_amp_flag = peak>min_amplitude
    
    # Eliminate Bad Pixel Sources
    bad_src_flag = np.ones((rows,), dtype=bool)
    for flags in badpix_flags:
        if isinstance(flags, six.string_types):
            flags = sources[flags]<=max_flag
        bad_src_flag = bad_src_flag & flags
    
    # Cut out elliptical sources (likely galaxies)
    if a is not None and b is not None:
        a_b = b/a
        if max_ratio is None:
            max_ratio = np.median(a_b)-np.std(a_b)*1.5
        elliptical_flag = b/a> max_ratio
    
    # Cut out all sources with a nearby neighbor
    if min_dist is None:
        min_dist = 3*aper_radius
    c0 = SkyCoord(ra, dec, unit=units)
    idx, d2, d3 = c0.match_to_catalog_sky(c0, 2)
    px = apu.def_unit('px', apu.arcsec*.27)
    d2 = d2.to(px).value
    distance_flag = d2>min_dist
    
    # Cut out all source near the edge
    if edge_dist is None:
        edge_dist = aper_radius+1
    edge_flag = ((x>edge_dist) & (y>edge_dist) &
        (x<img_data.shape[1]-edge_dist) & (y<img_data.shape[0]-edge_dist))
    
    # Apply all of the cuts to the table of sources
    psf_sources = sources[
        min_flux_flag & 
        min_amp_flag &
        bad_src_flag &
        elliptical_flag &
        distance_flag &
        edge_flag
    ]
    
    # Combine the flags into a table that can be used later
    flags = table.Table(
        [min_flux_flag, min_amp_flag, bad_src_flag, 
            elliptical_flag, distance_flag, edge_flag],
        names=('min_flux', 'min_amp', 'bad_pix', 'ellipse', 'dist', 'edge')
    )
    
    # If verbose, print information about the cuts
    if verbose:
        level = logger.getEffectiveLevel()
        logger.setLevel(logging.INFO)
        logger.info('Total sources: {0}'.format(len(sources)))
        logger.info('Sources with low flux: {0}'.format(
            len(sources)-np.sum(min_flux_flag)))
        logger.info('Sources with low amplitude: {0}'.format(
            len(sources)-np.sum(min_amp_flag)))
        logger.info('Sources with bad pixels: {0}'.format(
            len(sources)-np.sum(bad_src_flag)))
        logger.info('Elliptical sources: {0}'.format(
            len(sources)-np.sum(elliptical_flag)))
        logger.info('Source with close neighbors: {0}'.format(
            len(sources)-np.sum(distance_flag)))
        logger.info('Sources near an edge: {0}'.format(
            len(sources)-np.sum(edge_flag)))
        logger.info('Sources after cuts: {0}'.format(len(psf_sources)))
        logger.setLevel(level)
    return psf_sources, flags

def build_psf(img_data, aper_radius, sources=None, x='x', y='y', 
        subsampling=5, combine_mode='median', offset_buffer=3):
    """
    Build a subsampled psf by stacking a list of psf sources and
    a its source image.
    
    Parameters
    ----------
    img_data: array-like
        Image containing the sources
    aper_radius: integer
        Radius of the aperture to use for the psf source. This must be an
        odd number.
    sources: `~astropy.table.Table`, optional
        Table with x,y positions of psf sources. If sources is passed to the
        function, ``x`` and ``y`` should be the names of the x and y
        coordinates in the table.
    x: string or array-like, optional
        ``x`` can either be a string, the name of the x column in ``sources``,
        or an array of x coordinates of psf sources. *Defaults to 'x'*
    y: string or array-like, optional
        ``y`` can either be a string, the name of the y column in ``sources``,
        or an array of y coordinates of psf sources. *Defaults to 'y'*
    subsampling: integer, optional
        Number of subdivided pixel in the psf for each image pixel.
        *Defaults value is 5*
    combine_mode: string, optional
        Method to combine psf sources to build psf. Can be either
        "median" or "mean". *Defaults to "median"*
    offset_buffer: integer
        Error in image pixels for the center of a given source.
        This allows the code some leeway to re-center an image
        based on its subpixels. *Default is 3*
    """
    if isinstance(x, six.string_types):
        x = sources[x]
    if isinstance(y, six.string_types):
        y = sources[y]
    if hasattr(x, 'shape'):
        rows = x.shape[0]
    else:
        rows = len(x)
    
    if combine_mode == 'median':
        combine = np.ma.median
    elif combine_mode == 'mean':
        combine = np.ma.mean
    else:
        raise Exception(
            "Invalid combine_mode, you must choose from ['median','mean']")
    
    # Load a patch centered on each PSF source and stack them to build the
    # final PSF
    src_width = 2*aper_radius+1
    all_subpixels = []
    obj_shape = (src_width, src_width)
    for n in range(rows):
        subpixels = get_subpixel_patch(img_data, (y[n],x[n]), obj_shape, 
            offset_buffer=offset_buffer, subpixels=subsampling)
        all_subpixels.append(subpixels)
    psf = combine(np.dstack(all_subpixels), axis=2)
    
    # Add a mask to hide values outside the aperture radius
    radius = aper_radius*subsampling
    ys, xs = psf.shape
    yc,xc = (np.array([ys,xs])-1)/2
    y,x = np.ogrid[-yc:ys-yc, -xc:xs-xc]
    mask = x**2+y**2<=radius**2
    circle_mask = np.ones(psf.shape)
    circle_mask[mask]=0
    circle_mask
    psf_array = np.ma.array(psf, mask=circle_mask)
    return psf_array

def get_com_adjustment(arr):
    """
    Get the center of mass of an array. If the center of mass is the
    center pixel, the result will be (0,0). Any other value is what must
    be added to a position to get the center of mass.
    """
    total = np.ma.sum(arr)
    x_radius = (arr.shape[1]-1)/2.
    y_radius = (arr.shape[0]-1)/2.
    X = np.arange(-x_radius, x_radius+1, 1)
    Y = np.arange(-y_radius, y_radius+1, 1)
    X, Y = np.meshgrid(X, Y)
    dx = np.ma.sum(X*arr)/total
    dy = np.ma.sum(Y*arr)/total
    return (dx,dy)

class SinglePSF(Fittable2DModel):
    """
    A discrete PSF model for a single source. Before creating a 
    SinglePSF model it is necessary to run `build_psf` create a
    ``psf_array`` that is used to calibrate the source by modifying
    its amplitude and (optionally) its position.
    
    Parameters
    ----------
    psf_array: `numpy.ndarray`
        PSF array using ``subsampling`` subpixels for each pixel in the image
    amplitude: float
        Amplitude of the psf function
    x_0: float
        X coordinate in the image
    y_0: float
        Y coordinate in the image
    subsampling: int, optional
        Number of subpixels used in the ``psf_array`` for each pixel in the 
        image
    fix_com: bool
        Whether or not to correct for a psf peak that is not at the
        center of mass
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, psf_array, amplitude, x_0, y_0, subsampling=5,
            fix_com=False):
        # Set parameters defined by the psf to None so that they show
        # up as class attributes. self.set_psf_array will set all
        # of these parameters automatically
        self._subpixel_width = None
        self._width = None
        self._radius = None
        self._subsampling = None
        self._psf_array = None
        
        self.fitter = LevMarLSQFitter()
        
        # The position must be adjusted slightly beause
        # the center of mass isn't necessarily at the center
        # of the psf
        if fix_com:
            self.com = get_com_adjustment(psf_array)
        else:
            self.com = (0,0)
        # By default only the amplitude is fit
        # This can be changed in the fit function
        constraints = {'fixed': {'x_0': True, 'y_0': True}}
        super(SinglePSF, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                          amplitude=amplitude, **constraints)
        self.set_psf_array(psf_array, subsampling=subsampling)
    @property
    def shape(self):
        """
        Shape of the PSF image.
        """
        return self._psf_array.shape
    
    def evaluate(self, X, Y, amplitude, x_0, y_0):
        """
        Evaluate the SinglePSF model.
        """
        x = X[0,:]+x_0
        y = Y[:,0]+y_0
        result = amplitude * self._psf_func(x,y)
        return result
    
    def fit(self, img_data, fit_position=False, pos_range=0):
        """
        Fit the PSF to the data.
        
        Parameters
        ----------
        img_data: array-like
            Image data containing the source to fit
        fit_position: bool
            Whether or not to fit the position. If ``fit_position=False``
            only the amplitude will be fit
        pos_range: float
            Maximum distance that the position is allowed to shift
            from ``x_0,y_0`` initial. This is most useful when fitting
            a group of sources where the code might try to significantly
            move one of the sources for the fit. The default range
            is 0, which does not set any bounds at all.
        """
        # Choose whether or not the position of the PSF can be moved
        if fit_position:
            self.x_0.fixed = False
            self.y_0.fixed = False
            if pos_range>0:
                self.x_0.bounds = (self.x_0-pos_range, self.x_0+pos_range)
                self.y_0.bounds = (self.y_0-pos_range, self.y_0+pos_range)
        else:
            self.x_0.fixed = True
            self.y_0.fixed = True
        
        # Extract sub array with data of interest
        self.reset_psf_func(self.x_0.value, self.y_0.value)
        position = (self.y_0.value, self.x_0.value)
        sub_array_data = get_subpixel_patch(img_data, position, 
            (self._width, self._width), 0, self._subsampling, False)

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        if (sub_array_data.shape == self.shape and
                not np.isnan(sub_array_data).any()):
            X = np.linspace(-self._radius, self._radius, 
                self._subpixel_width)
            Y = np.linspace(-self._radius, self._radius, 
                self._subpixel_width)
            X, Y = np.meshgrid(X, Y)
            m = self.fitter(self, X, Y, sub_array_data)
            self.amplitude = m.amplitude
            self.x_0 = m.x_0
            self.y_0 = m.y_0
            return m.amplitude.value
        else:
            return 0
    
    def set_psf_array(self, psf_array, x_0=None, y_0=None, subsampling=None):
        """
        If a change is made to the psf function, it must
        be updated here so that all of the derived parameters
        (_width, _radius, _psf_array) can be updated
        """
        if subsampling is not None:
            self._subsampling = subsampling
        
        # Set the width and radius of the psf
        self._subpixel_width = max(psf_array.shape[0], psf_array.shape[1])
        self._width = self._subpixel_width/self._subsampling
        self._radius = (self._width-1)/2.
        self._psf_array = psf_array
        self.reset_psf_func(x_0, y_0)
    
    def reset_psf_func(self, x_0=None, y_0=None):
        """
        Reset the interpolation function used to represent the
        psf.
        """
        try:
            from scipy import interpolate
        except ImportError:
            raise Exception("Scipy must be installed to use psf fitting")
        if x_0 is None:
            x_0 = self.x_0.value+self.com[0]
        if y_0 is None:
            y_0 = self.y_0.value+self.com[0]
        # Set the function to determine the psf (up to an amplitude)
        X = np.linspace(-self._radius, self._radius, 
            self._subpixel_width)+x_0
        Y = np.linspace(-self._radius, self._radius, 
            self._subpixel_width)+y_0
        self._psf_func = interpolate.RectBivariateSpline(X, Y, self._psf_array)

class GroupPSF:
    """
    This represents the PSFs of a group of sources. In general
    a `GroupPSF` should only be created by the `ImagePSF` class.
    """
    def __init__(self, group_id, psf, positions=None, psf_width=None, 
            patch_boundaries=None, mask_img=True, 
            show_plots=True, **kwargs):
        if isinstance(psf, GroupPSF):
            self.__dict__ = psf.__dict__.copy()
        else:
            self.group_id = group_id
            self.psf = psf.copy()
            if psf_width is None and hasattr(self.psf, '_prf_array'):
                psf_width = self.psf._prf_array[0][0].shape[0]
            if psf_width % 2==0:
                raise Exception("psf_width must be an odd number")
            self.positions = positions
            self.psf_width = psf_width
            self.mask_img = mask_img
            self.show_plots = show_plots
            self.patch_boundaries=patch_boundaries
            self.combined_psf = None
    def get_patch(self, data, positions=None, width=None, 
            patch_boundaries=None):
        """
        Given a list of positions, get the patch of the data that
        contains all of the positions and their PRF radii and mask
        out all of the other pixels (to prevent sources outside the
        group from polluting the fit).
    
        Parameters
        ----------
        data: ndarray
            Image array data
        positions: list or array (optional)
            List of positions to include in the patch. If no 
            positions are passed the function will use 
            ``GroupPSF.positions``.
        width: int (optional)
            Width (in pixels) of the PRF. This should be an odd 
            number equal to 2*prf_radius+1 and defaults to 
            ``GroupPSF.psf_width``
        patch_boundaries: list or array (optional)
            Boundaries of the data patch of the form 
            [ymin,ymax,xmin,xmax]
        """
        if positions is None:
            positions = np.array(self.positions)
        if width is None:
            width = self.psf_width
    
        # Extract the patch of data that includes all of the sources
        # and their psf radii
        if patch_boundaries is None:
            if self.patch_boundaries is None:
                x = positions[:,0]
                y = positions[:,1]
                radius = int((width-1)/2)
                xc = np.round(x).astype('int')
                yc = np.round(y).astype('int')
                self.boundaries = [
                    min(yc)-radius, # ymin
                    max(yc)+radius+1, #ymax
                    min(xc)-radius, #xmin
                    max(xc)+radius+1 #xmax
                ]
            patch_boundaries = self.boundaries
        ymin,ymax,xmin,xmax = patch_boundaries
        sub_data = data[ymin:ymax,xmin:xmax]
        
        # If the group is large enough, sources not contained 
        # in the group might be located in the same square patch, 
        # so we mask out everything outside of the radius the 
        # individual sources PSFs
        if self.mask_img:
            sub_data = np.ma.array(sub_data)
            mask = np.ones(data.shape, dtype='bool')
            mask_X = np.arange(data.shape[1])
            mask_Y = np.arange(data.shape[0])
            mask_X, mask_Y = np.meshgrid(mask_X, mask_Y)
            for xc,yc in zip(x,y):
                mask = mask & (
                    (mask_X-xc)**2+(mask_Y-yc)**2>=(radius)**2)
    
            sub_data.mask = mask[ymin:ymax,xmin:xmax]
            sub_data = sub_data.filled(0)
    
        # Optionally plot the mask and data patch
        if self.show_plots:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d.axes3d import Axes3D
            except ImportError:
                raise Exception(
                    "You must have matplotlib installed"
                    " to create plots")
            # Plot mask
            if self.mask_img:
                plt.imshow(mask[ymin:ymax,xmin:xmax])
    
            # Plot masked patch used for fit
            X = np.arange(0, sub_data.shape[1], 1)
            Y = np.arange(0, sub_data.shape[0], 1)
            X, Y = np.meshgrid(X, Y)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(X, Y, sub_data)
            plt.show()
        return sub_data, (xmin, xmax), (ymin, ymax)
    
    def fit(self, data, positions=None, psfs=None, psf_width=None,
            patch_boundaries=None, fitter=None):
        """
        Simultaneously fit all of the sources in a PSF group. This 
        functions makes a copy of the PSF for each source and creates 
        an astropy `astropy.models.CompoundModel` that is fit but the PSF's
        ``fitter`` function.
    
        Parameters
        ----------
        data: ndarray
            Image data array
        positions : List or array (optional)
            List of positions in pixel coordinates
            where to fit the PSF/PRF. Ex: [(0.0,0.0),(1.0,2.0), (10.3,-3.2)]
        psfs : list of PSFs (optional)
            It is possible to choose a different PSF for each model, 
            in which case ``psfs`` should have the same length as positions
        psf_width: int (optional)
            Width of the PRF arrays. If all of the ``prfs`` are
            `photutils.psf.GaussianPSF` models then this will have to be
            set, otherwise it will automatically be calculated
        patch_boundaries: list or array (optional)
            Boundaries of the data patch of the form 
            [ymin,ymax,xmin,xmax]
        """
        if positions is None:
            positions = np.array(self.positions)
        x = positions[:,0]
        y = positions[:,1]
        
        if len(positions)==1:
            self.psf.x_0, self.psf.y_0 = positions[0]
            indices = np.indices(data.shape)
            result = [self.psf.fit(data, indices)]
        else:
            if psfs is None:
                psfs = [self.psf.copy() for p in range(len(positions))]
            if psf_width is None:
                if self.psf_width is not None:
                    psf_width = self.psf_width
                else:
                    psf_width = psfs[0]._prf_array[0][0].shape[0]
            if fitter is None:
                if self.psf is not None:
                    fitter = self.psf.fitter
                else:
                    fitter = psfs[0].fitter
            sub_data, x_range, y_range = self.get_patch(
                data, positions, psf_width, patch_boundaries)
    
            # Created a CompountModel that is a combination of the individual PRF's
            combined_psf = None
            for x0, y0, single_psf in zip(x,y,psfs):
                single_psf.x_0 = x0
                single_psf.y_0 = y0
                if combined_psf is None:
                    combined_psf = single_psf
                else:
                    combined_psf += single_psf
            # Fit the combined PRF
            indices = np.indices(data.shape)
            x_fit, y_fit = np.meshgrid(
                np.arange(x_range[0],x_range[1], 1),
                np.arange(y_range[0],y_range[1], 1))
            self.combined_psf = fitter(combined_psf, x_fit, y_fit, sub_data)
    
            # Return the list of fluxes for all of the sources in the group 
            # and the combined PRF
            result = [getattr(self.combined_psf,'amplitude_'+str(n)).value 
                for n in range(len(x))]
        return result

class ImagePSF:
    """
    Collection of Groups and PSFs for an entire image
    """
    def __init__(self, positions=None, psf=None, separation=None,
            cluster_method='dbscan', psf_width=None, mask_img=True,
            show_plots=False, groups=[]):
        self.positions = positions
        self.psf = psf
        self.separation = separation
        self.cluster_method = cluster_method
        self.psf_width = psf_width
        self.mask_img = mask_img
        self.show_plots = show_plots
        self.groups = groups
        self.group_indices = range(len(self.groups))
    
    def create_groups(self, positions=None, separation=None, 
            cluster_method='dbscan'):
        """
        Group sources with overlapping PSF's
        """
        if separation is None:
            if hasattr(self.psf, '_prf_array'):
                separation = self.psf._prf_array[0][0].shape[0]
            elif hasattr(self.psf, '_psf_array'):
                separation = (self.psf.psf_width-1/2.)
        if positions is None:
            positions = self.positions
        
        if cluster_method=='dbscan':
            # If user has sklearn installed, use DBSCAN to cluster the objects
            # in groups with overlapping PSF's
            try:
                from sklearn.cluster import DBSCAN
                from sklearn import metrics
                from sklearn.datasets.samples_generator import make_blobs
                from sklearn.preprocessing import StandardScaler
            except ImportError:
                Exception("You must install sklearn to use 'dbscan' clustering")
            
            
            pos_array = np.array(positions)
            # Compute DBSCAN
            db = DBSCAN(eps=separation, min_samples=1).fit(pos_array)
            self.db = db
            self.groups = []
            self.group_indices = np.unique(db.labels_)
            self.src_indices = db.labels_
            
        for group in self.group_indices:
            # Create PSF object for the entire group
            group_psf = GroupPSF(
                group, self.psf, pos_array[group==self.src_indices], 
                self.psf_width, mask_img=self.mask_img, 
                show_plots=self.show_plots)
            self.groups.append(group_psf)
        if self.show_plots:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except ImportError:
                raise Exception(
                    "You must have matplotlib installed to create plots")
            fig, ax = plt.subplots()
            x = pos_array[:,0]
            y = pos_array[:,1]
            for group in self.group_indices:
                ax.plot(
                    x[self.src_indices==group], 
                    y[self.src_indices==group], 'o')
            plt.show()
        return self.groups
    
    def get_psf_photometry(self, data, positions=None, psfs=None,
            separation=None, group_sources=True):
        if positions is None:
            if self.positions is None:
                raise Exception("You must supply a list of positions")
            positions = self.positions
                
        if group_sources:
            self.create_groups(positions)
        self.psf_flux = np.zeros(len(positions))
        pos_array = np.array(positions)
        for group in self.groups:
            fit_parameters = {
                'positions': pos_array[self.src_indices==group.group_id]
            }
            if psfs is not None:
                fit_parameters['psfs'] = psfs[self.src_indices==group.group_id]
            group_flux = group.fit(data,**fit_parameters)
            self.psf_flux[self.src_indices==group.group_id] = np.array(group_flux)
        return self.psf_flux