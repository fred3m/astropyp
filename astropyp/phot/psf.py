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
    #print 'normalized max', np.max(obj_data)
    return obj_data
#def get_psf_sources
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

#def get_prf
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

class SinglePSF(Fittable2DModel):
    """
    A discrete PSF model.
    
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
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, psf_array, amplitude, x_0, y_0, subsampling=5):
        self.fitter = LevMarLSQFitter()
        self._psf_array = psf_array
        self.subsampling = subsampling
        # By default only the amplitude is fit
        # This can be changed in the fit function
        constraints = {'fixed': {'x_0': True, 'y_0': True}}
        super(SinglePSF, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                          amplitude=amplitude, **constraints)
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
    
    def fit(self, img_data, fit_position=False):
        """
        Fit the PSF to the data.
        
        Parameters
        ----------
        img_data: array-like
            Image data containing the source to fit
        fit_position: bool
            Whether or not to fit the position. If ``fit_position=False``
            only the amplitude will be fit
        """
        from scipy import interpolate
        # Choose whether or not the position of the PSF can be moved
        if fit_position:
            self.x_0.fixed = False
            self.y_0.fixed = False
        else:
            self.x_0.fixed = True
            self.y_0.fixed = True
        
        # Set the function to determine the psf (up to an amplitude)
        x_radius = (self._psf_array.shape[1]-1)/2.
        y_radius = (self._psf_array.shape[0]-1)/2.
        X = np.arange(-x_radius, x_radius+1, 1)+self.x_0
        Y = np.arange(-y_radius, y_radius+1, 1)+self.y_0
        self._psf_func = interpolate.RectBivariateSpline(X, Y, self._psf_array)
        
        # Extract sub array with data of interest
        position = (self.y_0.value, self.x_0.value)
        src_width = np.array(self._psf_array.shape)/self.subsampling
        sub_array_data = get_subpixel_patch(img_data, position, 
            src_width, 0, self.subsampling, False)

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        if (sub_array_data.shape == self.shape and
                not np.isnan(sub_array_data).any()):
            #y = extract_array(indices[0], self.shape, position)
            #x = extract_array(indices[1], self.shape, position)
            X = np.arange(-x_radius, x_radius+1, 1)
            Y = np.arange(-y_radius, y_radius+1, 1)
            X, Y = np.meshgrid(X, Y)
            m = self.fitter(self, X, Y, sub_array_data)
            self.amplitude = m.amplitude
            self.x_0 = m.x_0
            self.y_0 = m.y_0
            return m.amplitude.value
        else:
            return 0