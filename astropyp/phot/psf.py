from __future__ import division
import numpy as np
import logging
from collections import OrderedDict

from astropy.extern import six
from astropy.nddata.utils import extract_array, subpixel_indices
from astropy.coordinates import SkyCoord
import astropy.units as apu
from astropy import table
from astropy.modeling import Fittable2DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling.fitting import LevMarLSQFitter

logger = logging.getLogger('astropyp.phot.psf')

psf_flags = OrderedDict([
    (128, 'Bad Pixel'), # Bad pixels
    (64, 'Edge'), # Edge is closer than aperture radius
    (32, '<not used>'),
    (16, '<not used>'),
    (8, 'Low Peak'), # Amplitude is below minimum
    (4, 'Low Flux'), # Flux is below minimum
    (2, 'Elliptical'), # Source is elliptical, not circular
    (1, 'Crowded') # nearby source, requires group photometry
])

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
        value is 1. *Default=True*
    
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
    data_func = interpolate.RectBivariateSpline(Y, X, data)
    X = np.arange(0, data.shape[1]*subpixels, 1)
    Y = np.arange(0, data.shape[0]*subpixels, 1)
    Z = data_func(Y, X, data)
    
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
        obj_data = Z
        if offset_buffer!=0:
            logger.warn(
                'Unable to re-center, maximum is at'
                ' {0} but offset buffer is {1}'.format(
                dpeak, offset_buffer))
    
    # Normalize the data so that each source will be on the same scale
    if normalize:
        obj_data = obj_data/np.max(obj_data)
    return obj_data

def select_psf_sources(img_data, catalog, aper_radius=None,
        min_flux=None, min_amplitude=None, min_dist=None, 
        max_ratio=None, edge_dist=None, verbose=True, units='deg', 
        badpix_flags=[], flag_max=0):
    """
    From an image and a list of sources, generate a list of high SNR
    sources with circular profiles that are likely to be good sources
    to generate the psf.
    
    Parameters
    ----------
    img_data: array-like
        The image data
    catalog: `~astropyp.catalog.ImageCatalog`
        Catalog of sources. The properties 
        x, y, ra, dec, aper_flux, peak, a, b 
        are all required to be mapped to the appropriate columns in the
        catalog.sources table. 
    aper_radius: integer
        Radius to use to calculate psf flux.
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
    units: string or `~astropy.units.unit`
        Units of the ra and dec columns. *Default value is 'deg'*
    badpix_flags: list of strings or list of array-like, optional
        A list of column names in ``sources`` or a list of arrays
        with the badpixel flags for each source.
        *Default is an empty list*
    flag_max: integer, optional
        Maximum value of a bad pixel flag to be considered good.
        *Default value is 0*
    
    Returns
    -------
    psf_idx: `~nump.ndarray`
        Boolean array that matches psf sources selected by the function
    flags: `~astropy.table.Table`
        Table of flags for all of the sources
    """
    rows = catalog.shape[0]
    
    # Only select high signal to noise sources
    min_flux_flag = ~np.isnan(catalog.aper_flux)
    if min_flux is not None:
        min_flux_flag = min_flux_flag & (catalog.aper_flux>min_flux)
    if min_amplitude is not None:
        min_amp_flag = catalog.peak>min_amplitude
    
    # Eliminate Bad Pixel Sources
    bad_src_flag = np.ones((rows,), dtype=bool)
    for flags in badpix_flags:
        if isinstance(flags, six.string_types):
            flags = catalog[flags]<=flag_max
        bad_src_flag = bad_src_flag & flags
    
    # Cut out elliptical sources (likely galaxies)
    if catalog.a is not None and catalog.b is not None:
        a_b = catalog.b/catalog.a
        if max_ratio is None:
            max_ratio = np.median(a_b)-np.std(a_b)*1.5
        elliptical_flag = catalog.b/catalog.a> max_ratio
    
    # Cut out all sources with a nearby neighbor
    if min_dist is None:
        min_dist = 3*aper_radius
    c0 = SkyCoord(np.array(catalog.ra), np.array(catalog.dec), unit=units)
    idx, d2, d3 = c0.match_to_catalog_sky(c0, 2)
    px = apu.def_unit('px', apu.arcsec*.27)
    d2 = d2.to(px).value
    distance_flag = d2>min_dist
    
    # Cut out all source near the edge
    if edge_dist is None:
        edge_dist = aper_radius+1
    edge_flag = ((catalog.x>edge_dist) & (catalog.y>edge_dist) &
        (catalog.x<img_data.shape[1]-edge_dist) & 
        (catalog.y<img_data.shape[0]-edge_dist))
    
    # Apply all of the cuts to the table of sources
    psf_idx = (
        min_flux_flag & 
        min_amp_flag &
        bad_src_flag &
        elliptical_flag &
        distance_flag &
        edge_flag)
    
    flags = (
        ~bad_src_flag*128+
        ~edge_flag*64+
        ~min_flux_flag*4+
        ~elliptical_flag*2+
        ~distance_flag)
    catalog.sources['pipeline_flags'] = flags
    
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
        logger.info('Total sources: {0}'.format(rows)),
        logger.info('Sources with low flux: {0}'.format(
            rows-np.sum(min_flux_flag)))
        logger.info('Sources with low amplitude: {0}'.format(
            rows-np.sum(min_amp_flag)))
        logger.info('Sources with bad pixels: {0}'.format(
            rows-np.sum(bad_src_flag)))
        logger.info('Elliptical sources: {0}'.format(
            rows-np.sum(elliptical_flag)))
        logger.info('Source with close neighbors: {0}'.format(
            rows-np.sum(distance_flag)))
        logger.info('Sources near an edge: {0}'.format(
            rows-np.sum(edge_flag)))
        logger.info('Sources after cuts: {0}'.format(np.sum(psf_idx))),
        logger.setLevel(level)
    return psf_idx, flags

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
        center of mass *Default=False*
    fit_position: bool
        Whether or not to fit the positions and the amplitude or only the 
        amplitude. *Default=True, fit positions and amplitude*
    pos_range: float
        +- bounds for the position (if ``fit_position=True``).
        If ``pos_range=0`` then no bounds are used and the 
        x_0 and y_0 parameters are free. *Default=0*
    """
    amplitude = Parameter('amplitude')
    x_0 = Parameter('x_0')
    y_0 = Parameter('y_0')

    def __init__(self, psf_array, amplitude, x_0, y_0, subsampling=5,
            fix_com=False, fit_position=True, pos_range=0):
        # Set parameters defined by the psf to None so that they show
        # up as class attributes. self.set_psf_array will set all
        # of these parameters automatically
        self._subpixel_width = None
        self._width = None
        self._radius = None
        self._subsampling = None
        self._psf_array = None
        
        self.fix_com = fix_com
        self.fitter = LevMarLSQFitter()
        
        # The position must be adjusted slightly beause
        # the center of mass isn't necessarily at the center
        # of the psf
        if fix_com:
            self.com = get_com_adjustment(psf_array)
        else:
            self.com = (0,0)
        super(SinglePSF, self).__init__(n_models=1, x_0=x_0, y_0=y_0,
                                          amplitude=amplitude)
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
        self.set_psf_array(psf_array, subsampling=subsampling)
    @property
    def shape(self):
        """
        Shape of the PSF image.
        """
        return self._psf_array.shape
    
    def get_flux(self, amplitude=None):
        """
        PSF Flux of the source
        """
        if amplitude is None:
            amplitude = self.amplitude
        flux = np.sum(self._psf_array*amplitude)/self._subsampling**2
        return flux
    
    def evaluate(self, X, Y, amplitude, x_0, y_0):
        """
        Evaluate the SinglePSF model.
        """
        x = X[0,:]-x_0
        y = Y[:,0]-y_0
        result = amplitude * self._psf_func(y,x)
        return result
    
    def fit(self, img_data, fit_position=True, pos_range=0, indices=None):
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
        indices: list of arrays
            ``indices`` is a list that contains the X and Y indices
            to the img_data provided. The default is None, which 
            uses the image scale and size of the psf to set the indices.
        """
        if indices is None:
            X = np.linspace(-self._radius, self._radius, 
                self._subpixel_width)+self.x_0-self.com[0]
            Y = np.linspace(-self._radius, self._radius, 
                self._subpixel_width)+self.y_0-self.com[1]
            X, Y = np.meshgrid(X, Y)
        else:
            X,Y = indices
        
        # Extract sub array with data of interest
        position = (self.y_0.value, self.x_0.value)
        sub_array_data = get_subpixel_patch(img_data, position, 
            (self._width, self._width), 0, self._subsampling, False)

        # Fit only if PSF is completely contained in the image and no NaN
        # values are present
        if (sub_array_data.shape == self.shape and
                not np.isnan(sub_array_data).any()):
            m = self.fitter(self, X, Y, sub_array_data)
            self.amplitude = m.amplitude
            self.x_0 = m.x_0
            self.y_0 = m.y_0
            return m.amplitude.value
        else:
            return 0
    
    def set_psf_array(self, psf_array, subsampling=None):
        """
        If a change is made to the psf function, it must
        be updated here so that all of the derived parameters
        (_width, _radius, _psf_array) can be updated
        """
        try:
            from scipy import interpolate
        except ImportError:
            raise Exception(
                "You must have scipy installed to use the SinglePSF class")
        if subsampling is not None:
            self._subsampling = subsampling
        
        # Set the width and radius of the psf
        self._subpixel_width = max(psf_array.shape[0], psf_array.shape[1])
        self._width = self._subpixel_width/self._subsampling
        self._radius = (self._width-1)/2.
        self._psf_array = psf_array
        
        # Set the function to determine the psf (up to an amplitude)
        X = np.linspace(-self._radius, self._radius, 
            self._subpixel_width)
        Y = np.linspace(-self._radius, self._radius, 
            self._subpixel_width)
        self._psf_func = interpolate.RectBivariateSpline(Y, X, self._psf_array)

class GroupPSF:
    """
    This represents the PSFs of a group of sources. In general
    a `GroupPSF` should only be created by the `ImagePSF` class.
    """
    def __init__(self, group_id, psf, positions=None, amplitudes=None,
            bounds=None, mask_img=True, show_plots=True, **kwargs):
        if isinstance(psf, GroupPSF):
            self.__dict__ = psf.__dict__.copy()
        else:
            self.group_id = group_id
            self.psf = psf.copy()
            self.positions = positions
            self.amplitudes = amplitudes
            self.mask_img = mask_img
            self.show_plots = show_plots
            self.bounds = bounds
            self.combined_psf = None
        for k,v in kwargs:
            setattr(self, k, v)
    def get_patch(self, data, positions=None, patch_bounds=None,
            show_plots=None):
        """
        Given a list of positions, get the patch of the data that
        contains all of the positions and their PSF radii and mask
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
        patch_bounds: list or array (optional)
            Boundaries of the data patch of the form 
            [ymin,ymax,xmin,xmax]
        """
        from scipy import interpolate
        
        if positions is None:
            positions = np.array(self.positions)
        if show_plots is None:
            show_plots = self.show_plots
        
        radius = int((self.psf._width-1)/2)
        x = positions[:,0]
        y = positions[:,1]
        
        # Extract the patch of data that includes all of the sources
        # and their psf radii
        if patch_bounds is None:
            if self.bounds is None:
                xc = np.round(x).astype('int')
                yc = np.round(y).astype('int')
                self.bounds = [
                    max(min(yc)-radius, 0), # ymin
                    min(max(yc)+radius+1, data.shape[0]), #ymax
                    max(min(xc)-radius, 0), #xmin
                    min(max(xc)+radius+1, data.shape[1]) #xmax
                ]
            patch_bounds = self.bounds
        ymin,ymax,xmin,xmax = patch_bounds
        X_img = np.arange(xmin, xmax+1, 1)
        Y_img = np.arange(ymin, ymax+1, 1)
        Z = data[np.ix_(Y_img,X_img)]
        X = np.arange(xmin, xmax+1, 1/self.psf._subsampling)
        Y = np.arange(ymin, ymax+1, 1/self.psf._subsampling)
        data_func = interpolate.RectBivariateSpline(Y_img, X_img, Z)
        sub_data = data_func(Y, X, Z)
        
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
            # Expand the mask to be the same size as the data patch
            subsampling = self.psf._subsampling
            sub_data.mask = np.kron(mask[np.ix_(Y_img,X_img)],
                np.ones((subsampling, subsampling), dtype=int))
            sub_data = sub_data.filled(0)
            
        # Optionally plot the mask and data patch
        if show_plots:
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
            #Xplt = np.arange(0, sub_data.shape[1], 1)
            #Yplt = np.arange(0, sub_data.shape[0], 1)
            Xplt, Yplt = np.meshgrid(X, Y)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_wireframe(Xplt, Yplt, sub_data, 
                rstride=5, cstride=5)
            plt.show()
        return sub_data, X, Y
    
    def fit(self, data, positions=None, psfs=None, amplitudes=None,
            patch_bounds=None, fitter=None):
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
        amplitudes: list of floats, optional
            Amplitudes (peak values) for the sources. If amplitudes is
            none then the pixel value at the position is used
        patch_bounds: list or array (optional)
            Boundaries of the data patch of the form 
            [ymin,ymax,xmin,xmax]
        fitter: `~astropy.modeling.Fitter`
            Fitting class to use to fit the data. *Default is self.fitter*
        """
        if positions is None:
            positions = np.array(self.positions)
            
        x = positions[:,0]
        y = positions[:,1]
        
        # If not estimated ampltiudes are given for the sources,
        # use the pixel value at their positions
        if amplitudes is None:
            amplitudes = data[y.astype(int),x.astype(int)]
        
        if len(positions)==1:
            self.psf.x_0, self.psf.y_0 = positions[0]
            result = [self.psf.fit(data)]
        else:
            if psfs is None:
                psfs = [self.psf.copy() for p in range(len(positions))]
            if fitter is None:
                if self.psf is not None:
                    fitter = self.psf.fitter
                else:
                    fitter = psfs[0].fitter
            sub_data, X, Y = self.get_patch(data, positions, patch_bounds)
            X,Y = np.meshgrid(X,Y)
    
            # Create a CompoundModel that is a combination of the 
            # individual PRF's
            combined_psf = None
            for x0, y0, single_psf, amplitude in zip(x,y,psfs,amplitudes):
                single_psf.x_0 = x0
                single_psf.y_0 = y0
                single_psf.amplitude = amplitude
                if combined_psf is None:
                    combined_psf = single_psf
                else:
                    combined_psf += single_psf
            # Fit the combined PRF
            self.combined_psf = fitter(combined_psf, X, Y, sub_data)
    
            # Return the list of fluxes for all of the sources in the group 
            # and the combined PRF
            result = [getattr(self.combined_psf,'amplitude_'+str(n)).value 
                for n in range(len(x))]
        return result