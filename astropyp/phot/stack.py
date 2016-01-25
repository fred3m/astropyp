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
        ccds: list of ccd names or list of ccds or `SingleImage`, optional
            CCDs to detect sources. This can also be ``self.stack``, which is
            the co-added `SingleImage` created when the individual images
            are stacked. 
            *Default is to detect sources on all CCDs in the stack*
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
        from astropyp.phot.phot import SingleImage
        if ccds is None:
            ccds = self.ccds
        elif isinstance(ccds, SingleImage):
            ccds = [ccds]
        
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
            self.merged_catalog = catalog
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
    
    def stack_images(self, combine_method='mean', dqmask_min=0, bad_pix_val=1,
            buf=10, order=3, pool_size=None, slices=None, wcs=None):
        """
        Stack all of the images into a single co-added image. 
        """
        from astropyp.phot.phot import SingleImage
        from astropyp.phot.stack import stack_full_images
        
        if wcs is None:
            import astropy.wcs
            wcs = astropy.wcs.WCS(self.ccds[self.ref_index].header)
        else:
            wcs = None
        imgs = []
        dqmasks = []
        tx_solutions = []
        for n,ccd in enumerate(self.ccds):
            if slices is None:
                imgs.append(ccd.img)
                dqmasks.append(ccd.dqmask)
            else:
                imgs.append(ccd.img[slices])
                dqmasks.append(ccd.dqmask[slices])
            if n!= self.ref_index:
                tx_solutions.append(self.tx_solutions[(self.ref_index, n)])
            else:
                tx_solutions.append(None)
        stack, stack_dqmask, patches, wcs = stack_full_images(
            imgs, self.ref_index, tx_solutions, dqmasks, combine_method,
            dqmask_min, bad_pix_val, buf, order, pool_size, wcs=wcs)
        self.wcs = wcs
        
        stack_params = OrderedDict(
            [('gain', None), ('exptime', None),('aper_radius', None)])
        for param in stack_params:
            if not hasattr(self, param):
                same_val = True
                pval = None
                for ccd in self.ccds:
                    if hasattr(ccd, param):
                        if pval is None:
                            pval = getattr(ccd, param)
                        elif pval!=getattr(ccd, param):
                            same_val = False
                            continue
                    else:
                        same_val = False
                        continue
                if same_val:
                    setattr(self, param, pval)
                    stack_params[param] = pval
        
        self.stack = SingleImage(img=stack.filled(0),
            dqmask=stack_dqmask, gain=stack_params['gain'], 
            exptime=stack_params['exptime'], 
            aper_radius=stack_params['aper_radius'])
        return stack, stack_dqmask

from astropyp.utils.misc import trace_unhandled_exceptions
@trace_unhandled_exceptions
def _init_full_stack(img_func, dqmask_func):
    """
    Initialize a process by storing the interpolating functions
    as global variables
    """
    import multiprocessing
    global gbl_img_func
    global gbl_dqmask_func
    
    gbl_img_func = img_func
    gbl_dqmask_func = dqmask_func
    
    logger.info("Initializing process {0}".format(
        multiprocessing.current_process().name))
    return

@trace_unhandled_exceptions
def _stack_worker(args):
    """
    Reproject a subset of an image or dqmask.
    """
    x,y = args
    img_data = gbl_img_func(y, x, grid=False)
    if gbl_dqmask_func is not None:
        dqmask_data = gbl_dqmask_func(zip(y,x))
    else:
        dqmask_data = None
    return (img_data, dqmask_data)

def stack_full_images(imgs, ref_index, tx_solutions, dqmasks = None,
            combine_method='mean', dqmask_min=0, bad_pix_val=1,
            buf=10, order=3, pool_size=None, wcs=None):
    """
    Combine a set of images into a stack using a full set
    of images using multiple processors (optional).
    """
    from scipy import interpolate
    from astropy.nddata import extract_array, overlap_slices
    import astropy.wcs
    from astropyp.astrometry import ImageSolution
    import multiprocessing
    
    buf = float(buf)
    if pool_size is None:
        pool_size = multiprocessing.cpu_count()
    elif pool_size==0:
        raise ValueError("pool_size must either be an integer>=1 "
            "or None, which sets the number of pools to the number "
            "of processors")
    
    # Get the minimum size of the final stack by projecting all of
    # the images onto the reference frame
    img_x = np.arange(0, imgs[ref_index].shape[1], 1)
    img_y = np.arange(0, imgs[ref_index].shape[0], 1)
    xmin = img_x[0]-buf
    xmax = img_x[-1]+buf
    ymin = img_y[0]-buf
    ymax = img_y[-1]+buf
    for n in range(len(imgs)):
        if n!=ref_index:
            tx_x,tx_y = tx_solutions[n].transform_coords(
                x=[img_x[0], img_x[-1]],
                y=[img_y[0], img_y[-1]])
            if tx_x[0]<xmin:
                xmin = tx_x[0]
            if tx_x[1]>xmax:
                xmax = tx_x[1]
            if tx_y[0]<ymin:
                ymin = tx_y[0]
            if tx_y[1]>ymax:
                ymax = tx_y[1]
    # Offset the referene image onto the coadd frame
    x_tx = OrderedDict([('Intercept', xmin), ('A_1_0', 1.0), ('A_0_1', 0.0)])
    y_tx = OrderedDict([('Intercept', ymin), ('B_1_0', 1.0), ('B_0_1', 0.0)])
    
    # Create a rough WCS system for the image (to be used later for
    # generating a better astrometric solution)
    if wcs is not None:
        wcs.wcs.crpix = [wcs.wcs.crpix[0]-xmin, wcs.wcs.crpix[1]-ymin]
    else:
        wcs = None
    
    # Modify the tx solutions to fit the coadd
    for n in range(len(imgs)):
        if n!= ref_index:
            new_x_tx = tx_solutions[n].x_tx.copy()
            new_y_tx = tx_solutions[n].y_tx.copy()
            new_x_tx['Intercept'] += xmin
            new_y_tx['Intercept'] += ymin
            tx_solutions[n] = ImageSolution(x_tx=new_x_tx, y_tx=new_y_tx,
                order=tx_solutions[n].order)
        else:
            tx_solutions[n] = ImageSolution(x_tx=x_tx, y_tx=y_tx, order=1)
    
    coadd_x = np.arange(0, xmax-xmin, 1)
    coadd_y = np.arange(0, ymax-ymin, 1)
    Xc, Yc = np.meshgrid(coadd_x, coadd_y)
    patches = []
    # Reproject each image to the coadded image
    for n in range(len(imgs)):
        tx_x,tx_y = tx_solutions[n].transform_coords(
            x=Xc.flatten(),y=Yc.flatten())
        tx_x = np.array(tx_x).reshape(Xc.shape)
        tx_y = np.array(tx_y).reshape(Yc.shape)
        img_x = np.arange(0, imgs[n].shape[1], 1)
        img_y = np.arange(0, imgs[n].shape[0], 1)
        # Create an interpolating function and dqmask interpolating function
        img_func = interpolate.RectBivariateSpline(img_y, img_x, 
            imgs[n],kx=order,ky=order)
        if dqmasks is not None:
            points = (img_y, img_x)
            values = dqmasks[n]
            dqmask_func = interpolate.RegularGridInterpolator(
                points, values, method='nearest', 
                fill_value=bad_pix_val, bounds_error=False)
        else:
            dqmask_func = None
        # If using a single processor, reproject the image and dqmask
        # in one iteration
        if pool_size==1:
            patch = img_func(tx_y.flatten(), tx_x.flatten(), grid=False)
            patch = patch.reshape(Xc.shape)
            #Create the dqmask
            if dqmasks is not None:
                dqmask = dqmask_func(zip(tx_y.flatten(),tx_x.flatten()))
                dqmask = dqmask.reshape(Xc.shape)
                dqmask = dqmask.astype(bool)
            else:
                dqmask = None
        # For multiple processors reproject line by line and recombine
        # once they are all completed
        else:
            logger.info('Processors:{0}'.format(pool_size))
            pool = multiprocessing.Pool(
                processes=pool_size,
                initializer=_init_full_stack,
                initargs=(img_func, dqmask_func))
            pool_args = []
            for m in range(tx_x.shape[0]):
                pool_args.append((tx_x[m], tx_y[m]))
            result = pool.map(_stack_worker, pool_args)
            pool.close()
            pool.join()
            patch, dqmask = zip(*result)
            patch = np.array(patch)
            dqmask = np.array(dqmask)
        if dqmask is not None:
            # Apply the dqmask to the image
            patch[dqmask>dqmask_min] = np.nan
            patch[dqmask>dqmask_min] 
            patch = np.ma.array(patch)
            patch.mask = np.isnan(patch)
        
        patches.append(patch)
    # Stack the reprojected images and create a dqmask and wtmap
    stack = np.ma.mean(patches, axis=0)
    dqmask = patches[0].mask
    wtmap = np.zeros(stack.shape)
    weight = 1/len(patches)
    for n in range(1,len(patches)):
        dqmask = np.bitwise_and(dqmask, patches[n].mask)
        wt = weight * ~patches[n].mask
        wtmap += wt
    dqmask = np.array(dqmask).astype('uint8')
    return stack, dqmask, wtmap, wcs