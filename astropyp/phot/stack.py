import numpy as np
import logging
from collections import OrderedDict
import multiprocessing

from astropyp.phot import phot

logger = logging.getLogger('astropyp.phot.stack')

class Stack(phot.SingleImage):
    """
    Images to stack together
    
    Parameters
    ----------
    ccds: list of `~astropyp.phot.phot.SingleImage`
        List of SingleImage ccd's.
    ref_index: static
        If ccds is a list this should be the index in the list of the
        ccd to use as a destination for the other re-projections. If no 
        ref_index is given (``ref_index=None``)
        then the init function will use the middle image
        as the reference.
    
    """
    def __init__(self, ccds, ref_index=None, **kwargs):
        if isinstance(ccds, list):
            if ref_index is None:
                ref_index = int(0.5*len(ccds))
        else:
            raise TypeError("ccds should be a list")
        self.ccd_count = len(ccds)
        self.ccd_indices = range(self.ccd_count)
        self.ccds = ccds
        self.ref_index = ref_index
        self.catalog = None
        
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def detect_sources(self, ccds=None, min_flux=None, psf_amplitude=None,
            good_amplitude=None, calibrate_amplitude=None, **kwargs):
        """
        Detect sources for all CCDs, or a subset of CCDs, in the stack.
        
        Parameters
        ----------
        ccds: list of ccd names or list of ccds, optional
            CCDs to detect sources. *Default is to detect sources
            on all CCDs in the stack*
        min_flux: int, optional
            Minimum flux needed for a source to be considered valid
        psf_amplitude: int, optional
            Minimum amplitude (in flux counts) for a source to be considered 
            a valid psf source.
        good_amplitude: int, optional
            Minimum amplitude for a source to be considered good.
            ``good_amplitude`` must either be specified for each ccd
            or in this function.
        calibrate_amplitude: int, optional
            Minimum amplitude for a source to be considered useful
            for calibration.
            ``calibrate_amplitude`` must either be specified for each ccd
            or in this function.
        kwargs: dict
            Keyword arguments to use in 
            ``~astropyp.SingleImage.detect_sources``
        """
        if ccds is None:
            ccds = self.ccds
        
        # Set the amplitudes used by various routines to cut on signal 
        # to noise
        amplitudes = OrderedDict([
            ('good', good_amplitude),
            ('calibrate', calibrate_amplitude), 
            ('psf', psf_amplitude)])
        for amp, amplitude in amplitudes.items():
            amp_name = amp+'_amplitude'
            if amplitude is None:
                if (not hasattr(self, amp_name) or 
                        getattr(self, amp_name) is None):
                    raise ValueError("You must specify a {0}".format(amp_name))
                amplitudes[amp] = getattr(self, amp_name)
            else:
                setattr(self, amp_name, amplitude)
        
        for ccd in ccds:
            # Save time by not getting windowed positions
            if 'windowed' not in kwargs.keys():
                kwargs['windowed'] = False
            ccd.detect_sources(**kwargs)
            ccd.select_psf_sources(min_flux, amplitudes['psf'])
            good_idx = ccd.catalog.sources['pipeline_flags']==0
            ccd.indices['good'] = good_idx & (
                ccd.catalog.sources['peak']>amplitudes['good'])
            ccd.indices['calibrate'] = good_idx & (
                ccd.catalog.sources['peak']>amplitudes['calibrate'])
        
    def merge_catalogs(self, good_indices=None, save_catalog=True):
        """
        Combine catalogs from each image into a master catalog that
        matches all of the source (by default, although specifying
        ``good_indices`` allows to only merge a subset of each
        catalog)
        
        Parameters
        ----------
        good_indices: string or list of arrays, optional
            If good_indices is a string, index ``ccd.indices[good_indices]``
            will be used for each CCD to filter the sources. Otherwise
            good_indices should be a list of indices to use for each CCD.
            If no god_indices are specified then all sources will be
            merged
        
        save_catalog: bool
            Whether or not to save the catalog
        
        Result
        ------
        catalog: `~astropyp.catalog.Catalog`
            Catalog of merged positions and indices for the sources
            in each image.
        """
        import astropyp.catalog
        from astropyp.utils.misc import update_ma_idx
        from astropy import table
        from astropy.extern import six
        
        if good_indices is None:
            all_ra = [ccd.catalog.ra for ccd in self.ccds]
            all_dec = [ccd.catalog.dec for ccd in self.ccds]
        else:
            if isinstance(good_indices, six.string_types):
                good_indices = [ccd.indices[good_indices] for ccd in self.ccds]
            all_ra = [ccd.catalog.ra[good_indices[n]] 
                for n, ccd in enumerate(self.ccds)]
            all_dec = [ccd.catalog.dec[good_indices[n]] 
                for n, ccd in enumerate(self.ccds)]
        indices, matched = astropyp.catalog.get_merged_indices(all_ra, all_dec)
        all_x = []
        all_y = []
        for n,ccd in enumerate(self.ccds):
            ccd.indices['merge'] = indices[n]
            if n!=self.ref_index:
                x0,y0 = self.tx_solutions[
                    (n,self.ref_index)].transform_coords(ccd.catalog)
            else:
                x0,y0 = (ccd.catalog.x, ccd.catalog.y)
            if good_indices is not None:
                x0 = x0[good_indices[n]]
                y0 = y0[good_indices[n]]
            all_x.append(x0)
            all_y.append(y0)
        all_x = [update_ma_idx(all_x[n], self.ccds[n].indices['merge'])
            for n in self.ccd_indices]
        all_y = [update_ma_idx(all_y[n], self.ccds[n].indices['merge'])
            for n in self.ccd_indices]
        sources = table.Table(masked=True)
        sources['x'],sources['y'] = astropyp.catalog.combine_coordinates(
            all_x, all_y, 'mean')
        for n,ccd in enumerate(self.ccds):
            x_col = 'x_{0}'.format(n)
            y_col = 'y_{0}'.format(n)
            sources[x_col] = all_x[n]
            sources[y_col] = all_y[n]
            if good_indices is None:
                src_idx = self.ccds[n].catalog['src_idx']
            else:
                src_idx = self.ccds[n].catalog['src_idx'][good_indices[n]]
            src_idx = update_ma_idx(src_idx, 
                self.ccds[n].indices['merge'])
            sources['idx_{0}'.format(n)] = src_idx
            x_diff = sources[x_col]-sources['x']
            y_diff = sources[y_col]-sources['y']
            logger.info('x rms: {0}'.format(
                np.sqrt(np.sum(x_diff)**2/len(sources))))
            logger.info('y rms: {0}'.format(
                np.sqrt(np.sum(y_diff)**2/len(sources))))
        catalog = astropyp.catalog.Catalog(sources)
        if save_catalog:
            self.catalog = catalog
        return catalog
        
    def get_transforms(self, ra_name='ra', dec_name='dec', match_kwargs={}):
        """
        Get transformations to convert all images to the coordinates
        of the reference images
        """
        import astropyp.catalog
        from astropyp import astrometry
        catalogs = []
        for ccd in self.ccds:
            sources = ccd.catalog.sources
            catalogs.append(sources[ccd.indices['calibrate']])
        catalogs = astropyp.catalog.match_all_catalogs(catalogs,
            ra_name, dec_name, combine=False, **match_kwargs)
        pairs = [(idx, self.ref_index) for idx in self.ccd_indices if
            idx!=self.ref_index]
        pairs += [(self.ref_index, idx) for idx in self.ccd_indices if
            idx!=self.ref_index]
        self.tx_solutions = {}
        for pair in pairs:
            idx1, idx2 = pair
            self.tx_solutions[pair] = astrometry.ImageSolution(order=1)
            self.tx_solutions[pair].get_solution(catalogs[idx1],catalogs[idx2])
            logger.info('{0}:{1}'.format(pair, self.tx_solutions[pair].stats))
    
    def create_psf(self, aper_radius=None, method='mean', 
            create_catalog=True, good_indices=None, save_catalog=False,
            single_thread=False, pool_size=None, subsampling=None,
            max_offset=0):
        """
        Create a PSF from the psf sources in the ccd catalog. By default this
        will use multiprocessing to stack the sources as quickly as possible
        but this can be disabled.
        
        Parameters
        ----------
        aper_radius: float, optional
            Aperture radius to use for the psf. The default is to use the
            aperture radius set for the stack, and if that hasn't been set
            the aperture radius for the individual CCD's.
        method: string, optional
            This should be either 'median' or 'mean', describing how the
            individual images should be combined into the psf.
            *Default is 'mean'*
        create_catalog: bool
            Whether or not to create a catalog of psf stars.
            *Default=True*
        good_indices: string or list of arrays, optional
            If good_indices is a string, index ``ccd.indices[good_indices]``
            will be used for each CCD to filter the sources. Otherwise
            good_indices should be a list of indices to use for each CCD.
            If no god_indices are specified then all sources will be
            merged
        
        save_catalog: bool
            Whether or not to save the catalog
        
        Result
        ------
        psf: `~numpy.ndarray`
            Subpixel array with the psf for the stack
        """
        from astropyp.phot.psf import SinglePSF
        if aper_radius is None:
            if not hasattr(self, 'aper_radius'):
                if not hasattr(self.ccds[0], 'aper_radius'):
                    raise ValueError("You must specify an aper_radius")
                else:
                    self.aper_radius = self.ccds[0].aper_radius
            aper_radius = self.aper_radius
        if subsampling is None:
            if not hasattr(self, 'subsampling'):
                self.subsampling = self.ccds[0].subsampling
            subsampling = self.subsampling
        
        if method=='median':
            combine = np.ma.median
        elif method=='mean':
            combine = np.ma.mean
        else:
            raise ValueError("Method must be either 'median' or 'mean'")
        
        src_shape = (2*aper_radius+1, 2*aper_radius+1)
        
        if create_catalog:
            catalog = self.merge_catalogs(good_indices, save_catalog)
        elif self.catalog is not None:
            catalog = self.catalog
        else:
            raise Exception("You must create a catalog to create a psf")
        
        imgs = []
        dqmasks = []
        tx_solutions = []
        for n,ccd in enumerate(self.ccds):
            imgs.append(ccd.img)
            dqmasks.append(ccd.dqmask)
            if n!= self.ref_index:
                tx_solutions.append(self.tx_solutions[(self.ref_index, n)])
            else:
                tx_solutions.append(None)

        # Select sources that pass PSF cuts in all images
        psf_sources = catalog.sources
        print('psf sources', len(psf_sources))
        idx = np.zeros((len(psf_sources),), dtype=bool)
        for col in ['idx_'+str(n) for n in self.ccd_indices]:
            idx = idx | psf_sources[col].mask
        psf_sources = psf_sources[~idx]
        print('used psf sources', len(psf_sources))
        if single_thread:
            patches = []
            for src in psf_sources:
                data, dqmask = stack_source(imgs, src['x'], src['y'], 
                    src_shape, self.ref_index, tx_solutions, 
                    dqmasks, combine_method='mean', show_plots=False,
                    subsampling=self.subsampling, max_offset=max_offset)
                data = data/np.max(data)
                patches.append(data)
        else:
            if pool_size is None:
                pool_size = multiprocessing.cpu_count()
                # Create a pool with the static (for all sources)
                # variables to speed up processing
                pool = multiprocessing.Pool(
                    initializer=_init_multiprocess,
                    initargs=(imgs, dqmasks, tx_solutions, src_shape,
                        self.ref_index, method, self.subsampling))
                src_coords = []
                for src in psf_sources:
                    src_coords.append((src['x'], src['y']))
                result = pool.map(_multi_stack_worker, src_coords)
                patches, dqmasks = zip(*result)
                patches = [p/np.max(p) for p in patches]
                pool.close()
                pool.join()
        psf_array = np.ma.array(patches)
        psf_array = combine(psf_array, axis=0)
        
        # Add a mask to hide values outside the aperture radius
        if not hasattr(self, 'subsampling'):
            self.subsampling = self.ccds[0].subsampling
        radius = aper_radius*self.subsampling
        ys, xs = psf_array.shape
        yc,xc = (np.array([ys,xs])-1)/2
        y,x = np.ogrid[-yc:ys-yc, -xc:xs-xc]
        mask = x**2+y**2<=radius**2
        circle_mask = np.ones(psf_array.shape)
        circle_mask[mask]=0
        circle_mask
        psf_array = np.ma.array(psf_array, mask=circle_mask)
        self.psf = SinglePSF(psf_array, 1., 0, 0, self.subsampling)
        return self.psf
    
    def perform_psf_photometry(self, separation=None, 
            verbose=False, fit_position=True, pos_range=0, indices=None,
            kd_tree=None, exptime=None, pool_size=None, stack_method='mean',
            save_catalog=True, single_thread=False):
        """
        Perform PSF photometry on all of the sources in the catalog,
        or if indices is specified, a subset of sources.
    
        Parameters
        ----------
        separation: float, optional
            Separation (in pixels) for members to be considered
            part of the same group *Default=1.5*psf width*
        verbose: bool, optional
            Whether or not to show info about the fit progress.
            *Default=False*
        fit_position: bool, optional
            Whether or not to fit the position along with the
            amplitude of each source. *Default=True*
        pos_range: int, optional
            Maximum number of pixels (in image pixels) that
            a sources position can be changed. If ``pos_range=0``
            no bounds will be set. *Default=0*
        indices: `~numpy.ndarray` or string, optional
            Indices for sources to calculate PSF photometry.
            It is often advantageous to remove sources with
            bad pixels and sublinear flux to save processing time.
            All sources not included in indices will have their
            psf flux set to NaN. This can either be an array of
            indices for the positions in self.catalog or 
            the name of a saved index in self.indices
        """
        import astropyp.catalog
        from astropyp.phot.psf import SinglePSF
        
        # If no exposure time was passed to the stack and all of the
        # ccds had the same exposure time, use that to calcualte the
        # psf magnitude
        if exptime is None:
            if hasattr(self, 'exptime'):
                exptime = self.exptime
            else:
                same_exptime = True
                exptime = np.nan
                for ccd in self.ccds:
                    if np.isnan(exptime):
                        exptime = ccd.exptime
                    elif ccd.exptime != exptime:
                        same_exptime = False
                if not same_exptime:
                    exptime = None
        
        if hasattr(self.catalog, 'peak'):
            peak = self.catalog.peak
        else:
            from astropyp.utils.misc import update_ma_idx
            peak = [
                update_ma_idx(self.ccds[n].catalog['peak'],
                self.catalog['idx_'+str(n)]) 
                    for n in self.ccd_indices]
            peak = np.ma.mean(peak, axis=0)
        
        # Get the positions and estimated amplitudes of
        # the sources to fit
        
        if indices is not None:
            positions = zip(self.catalog.x[indices], 
                            self.catalog.y[indices])
            amplitudes = peak[indices]
        else:
            positions = zip(self.catalog.x, self.catalog.y)
            amplitudes = peak
    
        src_count = len(positions)
    
        src_indices = np.arange(0,len(self.catalog.sources),1)
        all_positions = np.array(zip(self.catalog.x, self.catalog.y))
        all_amplitudes = peak
        total_sources = len(all_amplitudes)
        src_psfs = []
    
        psf_flux = np.ones((total_sources,))*np.nan
        psf_flux_err = np.ones((total_sources,))*np.nan
        psf_x = np.ones((total_sources,))*np.nan
        psf_y = np.ones((total_sources,))*np.nan
        new_amplitudes = np.ones((total_sources,))*np.nan
    
        # Find nearest neighbors to avoid contamination by nearby sources
        if separation is not None:
            if not hasattr(self, 'kdtree'):
                from scipy import spatial
                KDTree = spatial.cKDTree
                self.kd_tree = KDTree(all_positions)
            idx, nidx = astropyp.catalog.find_neighbors(separation, 
                kd_tree=self.kd_tree)
        # Get parameters to pass to the pool manager
        imgs = []
        dqmasks = []
        tx_solutions = []
        for n,ccd in enumerate(self.ccds):
            imgs.append(ccd.img)
            dqmasks.append(ccd.dqmask)
            if n!= self.ref_index:
                tx_solutions.append(self.tx_solutions[(self.ref_index, n)])
            else:
                tx_solutions.append(None)
        
        if pool_size is None:
            pool_size = multiprocessing.cpu_count()
        # Create a pool with the static (for all sources)
        # variables to speed up processing
        pool = multiprocessing.Pool(
            processes=pool_size,
            initializer=_init_multiprocess,
            initargs=(imgs, dqmasks, tx_solutions, self.psf.shape,
                self.ref_index, stack_method, self.subsampling, 
                self.psf))
        pool_args = []
        # Fit each source to the PSF, calcualte its flux, and its new
        # position
        for n in range(len(positions)):
            if indices is not None:
                src_idx = src_indices[indices][n]
            else:
                src_idx = src_indices[n]
            if separation is not None:
                n_indices = nidx[idx==src_idx]
                neighbor_positions = all_positions[n_indices]
                neighbor_amplitudes = all_amplitudes[n_indices]
            else:
                neighbor_positions = []
                neighbor_amplitudes = []
            pool_args.append((amplitudes[n], 
                (positions[n][1], positions[n][0]),
                self.subsampling,
                neighbor_positions,
                neighbor_amplitudes))
    
        result = pool.map(_stack_psf_worker, pool_args)
        pool.close()
        pool.join()
    
        # Update the various psf array parameters
        for n in range(len(positions)):
            if indices is not None:
                src_idx = src_indices[indices][n]
            else:
                src_idx = src_indices[n]
            psf_flux[src_idx] = result[n][0]
            psf_flux_err[src_idx] = result[n][1]
            new_amplitudes[src_idx] = result[n][2]
            psf_x[src_idx], psf_y[src_idx] = result[n][3]
    
        # Save the psf derived quantities in the catalog
        # Ignore divide by zero errors that occur when sources
        # have zero psf flux (i.e. bad sources)
        if save_catalog:
            np_err = np.geterr()
            np.seterr(divide='ignore')
            self.catalog.sources['psf_flux'] = psf_flux
            self.catalog.sources['psf_flux_err'] = psf_flux_err
            if exptime is not None:
                psf_mag = -2.5*np.log10(psf_flux/exptime)
                self.catalog.sources['psf_mag'] = psf_mag
                self.catalog.sources['psf_mag_err'] = psf_flux_err/psf_flux
            self.catalog.sources['psf_x'] = psf_x
            self.catalog.sources['psf_y'] = psf_y
            np.seterr(**np_err)
        
        return psf_flux, psf_flux_err

def _init_multiprocess(imgs, dqmasks, tx_solutions, src_shape,
        ref_index, combine_method, subsampling, psf=None):
    """
    This process is called each time a new process is initialized
    """
    global gbl_imgs
    global gbl_dqmasks
    global gbl_tx_solutions
    global gbl_src_shape
    global gbl_ref_index
    global gbl_combine_method
    global gbl_psf
    global gbl_subsampling
    
    gbl_imgs = imgs
    gbl_dqmasks = dqmasks
    gbl_tx_solutions = tx_solutions
    gbl_src_shape = src_shape
    gbl_ref_index = ref_index
    gbl_combine_method = combine_method
    gbl_subsampling = subsampling
    gbl_psf = psf
    logger.info("Multiprocessing Initialized")

import traceback, functools
def trace_unhandled_exceptions(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print 'Exception in '+func.__name__
            traceback.print_exc()
            raise Exception("Error in multiprocessing")
    return wrapped_func

@trace_unhandled_exceptions
def _multi_stack_worker(args):
    """
    Worker to stack sources using multiprocessing
    """
    x,y = args
    result = stack_source(gbl_imgs, x, y, 
        gbl_src_shape, gbl_ref_index, gbl_tx_solutions, 
        gbl_dqmasks, combine_method=gbl_combine_method, 
        show_plots=False)
    return result

@trace_unhandled_exceptions
def _stack_psf_worker(args):
    """
    Worker to stack each source and perform psf photometry
    on the stacked patch
    """
    from astropyp.phot.psf import SinglePSF
    amplitude = args[0]
    y0, x0 = args[1]
    subsampling = args[2]
    neighbor_positions = args[3]
    neighbor_amplitudes = args[4]
    
    # Stack images centered on the source
    patch, dqmask = stack_source(gbl_imgs, x0, y0, 
        gbl_src_shape, gbl_ref_index, gbl_tx_solutions, 
        gbl_dqmasks, combine_method=gbl_combine_method, 
        show_plots=False, subsampling=1)
    if patch is None:
        return np.nan, np.nan, np.nan, (y0, x0)
    # Perform PSF phot on the stacked source
    src_psf = SinglePSF(gbl_psf._psf_array, 
        amplitude, x0, y0,
        subsampling=1,
        neighbor_positions=neighbor_positions,
        neighbor_amplitudes=neighbor_amplitudes)
    x_radius = gbl_psf.shape[1]/gbl_subsampling
    y_radius = gbl_psf.shape[0]/gbl_subsampling
    X = np.linspace(x0-x_radius, x0+x_radius, gbl_psf.shape[1])
    Y = np.linspace(y0-y_radius, y0+y_radius, gbl_psf.shape[0])
    psf_flux, psf_flux_err, residual, new_pos = src_psf.fit(
        patch=patch, X=X, Y=Y)
    new_amplitude = src_psf.amplitude.value
    return psf_flux, psf_flux_err, src_psf.amplitude.value, new_pos

def reproject_image(img, x, y, new_x, new_y, tx_solution=None,
        dqmask=None, subsampling=5, 
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
    from astropy.nddata.utils import overlap_slices
    import astropyp.utils
    from astropyp.utils.misc import extract_array
    from scipy.ndimage import zoom
    
    src_shape = (len(new_y)/subsampling, len(new_x)/subsampling)
    
    # Convert the centroid position from destination coordinates
    # to current image coordinates and extract the subsampled image
    if tx_solution is not None:
        x0,y0 = tx_solution.transform_coords(x=x, y=y)
    else:
        x0,y0 = (x,y)
    
    data, slices = extract_array(img,
        src_shape, (y0,x0), subsampling=subsampling, return_slices=True)
    y_radius = int(src_shape[0]/2)
    x_radius = int(src_shape[1]/2)
    X0 = np.linspace(x0-x_radius, x0+x_radius, data.shape[1])
    Y0 = np.linspace(y0-y_radius, y0+y_radius, data.shape[0])
    
    #Z = interpolate.RectBivariateSpline(
    #    Y0[slices[1][0]], X0[slices[1][1]], data[slices[1]])
    X0 = X0[slices[1][1]]
    Y0 = Y0[slices[1][0]]
    Z = interpolate.RectBivariateSpline(Y0,X0, data[slices[1]])
    
    # Convert the coordinates from the destination to the
    # current image and extract the interpolated pixels
    if tx_solution is not None:
        X,Y = tx_solution.transform_coords(x=new_x, y=new_y)
    else:
        X,Y = (new_x, new_y)
    X = X[slices[1][1]]
    Y = Y[slices[1][0]]
    
    modified_data = np.zeros(data.shape)
    modified_data[:] = np.nan
    modified_data[slices[1]] = Z(Y,X)
    modified_data = np.ma.array(modified_data)
    modified_data.mask = np.isnan(modified_data)
    if dqmask is not None:
        modified_dqmask = np.zeros(data.shape)
        modified_dqmask[:] = bad_pix_val
        new_dqmask = extract_array(dqmask, src_shape, (y0,x0), 
            masking=True, fill_value=bad_pix_val)
        new_dqmask = zoom(new_dqmask, subsampling, order=0)
        new_dqmask = new_dqmask[slices[1]]
        Xold,Yold = np.meshgrid(X0,Y0)
        Xnew,Ynew = np.meshgrid(X,Y)
        coords = zip(Yold.flatten(), Xold.flatten())
        new_coords = zip(Ynew.flatten(), Xnew.flatten())
        new_dqmask = interpolate.griddata(coords,
            new_dqmask.flatten(),new_coords, method='nearest')
        modified_dqmask[slices[1]] = new_dqmask.reshape(Xold.shape)
        modified_data.mask = modified_data.mask | (modified_dqmask>dqmask_min)
    else:
        modified_dqmask = None
    if show_plot:
        import matplotlib
        import matplotlib.pyplot as plt
        plt.imshow(modified_data, interpolation='none')
        plt.show()
    return modified_data, modified_dqmask

def stack_source(images, x, y, src_shape, ref_index, tx_solutions,
        dqmasks=None, subsampling=5, combine_method='mean', 
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
    ref_index: int
        Index in ``images`` and ``dqmasks`` of the image the others
        are projected onto for the stack.
    tx_solutions: list of `~astropyp.catalog.ImageSolution`s
        Transformations to convert each image to the projected coordinates
    dqmasks: list of `~numpy.ndarrays`'s, optional
        Dqmasks for images to stack
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
    from astropyp.utils.misc import extract_array
    import astropyp.utils
    
    if combine_method=='median':
        combine = np.ma.median
    elif combine_method=='mean':
        combine = np.ma.mean
    else:
        raise Exception(
            "Combine method must be either 'median' or 'mean'")
    # Get the patch from the original image
    data = extract_array(images[ref_index],
        src_shape, (y,x), subsampling=subsampling)
    y_radius = src_shape[0]>>1
    x_radius = src_shape[1]>>1
    x_new = np.linspace(x-x_radius, x+x_radius, data.shape[1])
    y_new = np.linspace(y-y_radius, y+y_radius, data.shape[0])
    
    # Mask bad pixels and edges
    data = np.ma.array(data)
    data.mask = np.isnan(data)
    
    if dqmasks is not None:
        # Get the data quality mask for the original image and
        # set any edge values to the bad_pix_val, and scale the dqmask
        # with a nearest neighbor (pixelated) interpolation
        dqmask = extract_array(dqmasks[ref_index], src_shape, (y,x), 
                               fill_value=bad_pix_val)
        dqmask[dqmask<0] = bad_pix_val
        dqmask = zoom(dqmask, subsampling, order=0)
        data.mask = data.mask | (dqmask>dqmask_min)
        modified_dqmask = [None, dqmask, None]
    else:
        dqmask = None
        modified_dqmask = [None,None,None]
        dqmasks = [None,None,None]
    
    if show_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        if dqmask is not None:
            plt.imshow(dqmask, interpolation='none')
            plt.show()
        if data is not None:
            plt.imshow(data, interpolation='none')
            plt.show()
    # Reproject the images
    modified_data = [None, data, None]
    for n in [m for m in range(len(images)) if m!=ref_index]:
        modified_data[n],modified_dqmask[n] = reproject_image(
            images[n], x, y, x_new, y_new, tx_solutions[n], dqmasks[n],
            subsampling, show_plots, dqmask_min, bad_pix_val)
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