import numpy as np
import logging
from astropy import table
from astropy.nddata.utils import extract_array

logger = logging.getLogger('astropyp.phot.detect')

def get_img_flags(dqmask, x, y, shape, edge_val=1):
    """
    Parameters
    ----------
    dqmask: array-like
        Data quality mask for the image
    x: array-like
        x coordinates of points
    y: array-like
        y coordinates of points
    shape: tuple
        Shape of the patch to extract from the image
    edge_val: integer
        Value to use for pixels outside the edge of the image
        
    Returns
    -------
    img_flags: `~numpy.ndarray`
        Flags selected from the bad pixel mask
    """
    if hasattr(x, 'shape'):
        rows = x.shape[0]
    else:
        rows = len(x)
    img_flags = []
    for n in range(rows):
        patch = extract_array(dqmask, shape, (y[n], x[n]), fill_value=edge_val)
        val = 0
        for n in np.unique(patch):
            val = val | n
        img_flags.append(val)
    img_flags = np.array(img_flags)
    return img_flags

def get_sources(img_data, dqmask_data=None, wtmap_data=None, exptime=None, 
        sex_params={}, objects=None, subtract_bkg=False, gain=None, 
        wcs=None, aper_radius=None, windowed=True, edge_val=1, origin=0,
        transform='wcs'):
    """
    Load sources from an image and if a data quality mask is provided, included
    a bitmap of flags in the output catalog. This using SEP to detect
    sources using SExtractor functions (unless an input catalog is provided) 
    and by default calculates the windowed position (see SExtractor or SEP docs
    for more details).
    
    Parameters
    ----------
    img_data: array-like
        Image to use for search
    dqmask_data: array-like, optional
        Data quality mask array. If this is included the output catalog
        will include a ``flags`` column that gives a binary flag for bad
        pixels in the image.
    wtmap: array-like, optional
        Not currently implemented
    exptime: float, optional
        Exposure time. If ``exptime`` is not given, then no magnitudes
        will be calculated
    sex_params: dict, optional
        Dictionary of SExtractor parameters to pass to the
        `~sep.extract` function.
    objects: `~astropy.table.Table`, optional
        If a catalog of sources has already been generated, SExtraction
        will be skipped and the input catalog will be used for aperture
        photometry. The input catalog must contain a list of sources
        either ``ra,dec`` or ``x,y`` as columns.
    subtract_bkg: bool, optional
        Whether or not to subtract the background
    gain: float, optional
        Gain of the detector, used to calculate the magnitude error
    wcs: `~astropy.wcs.WCS`, optional
        World coordinates to calculate RA,DEC from image X,Y. If ``wcs``
        is not given then the output catalog will only contain cartesian
        coordinates for each source.
    aper_radius: int, optional
        Radius of the aperture to use for photometry. If no ``aperture_radius``
        is specified, only the Kron flux will be included in the final
        catalog
    windowed: bool, optional
        Whether or not to use SExtractors window algorithm to calculate a more
        precise position. *Default=True*
    edge_val: integer
        Value to use for pixels outside the edge of the image
    transform: string, optional
        Type of transform to use. Either 'wcs','sip', or 'all'. 
        *Default=wcs*
    """
    import sep
    
    # Estimate the background and subtract it (in place) from the image
    # Note: we might not need to do background subtraction due to the 
    # community pipeline
    bkg = sep.Background(img_data, dqmask_data)
    if subtract_bkg:
        bkg.subfrom(img_data)
        bkg = sep.Background(img_data, dqmask_data)
    
    # Find the objects
    if objects is None:
        if 'extract' not in sex_params:
            sex_params['extract'] = {'thresh': 1.5*bkg.globalrms}
        if 'thresh' not in sex_params['extract']:
            if 'thresh' in sex_params:
                sex_params['extract']['thresh'] = \
                    sex_params['thresh']*bkg.globalrms
            else:
                raise Exception(
                    "You must provide a threshold parameter"
                    " for source extraction")
        sources = sep.extract(img_data, **sex_params['extract'])
        objects = table.Table(sources)
        # Remove sources with a>>b (usually cosmic rays)
        #objects = objects[objects['a']<objects['b']*5]
    
    # Set WCS or X, Y if necessary
    if wcs is not None:
        if transform=='wcs':
            transform_method = wcs.wcs_pix2world
        elif transform=='all':
            transform_method = wcs.all_pix2world
        elif transform=='sip':
            transform_method = wcs.sip_pix2foc
    if 'ra' not in objects.columns.keys() and wcs is not None:
        objects['ra'], objects['dec'] = transform_method(
            objects['x'], objects['y'], 0)
    if 'x' not in objects.columns.keys():
        if wcs is None:
            raise Exception("You must provide a wcs transformation if "
                "specifying ra and dec")
        objects['x'], objects['y'] = wcs.all_world2pix(
            objects['ra'], objects['dec'], 0)
    
    if windowed:
        logger.info("using kron to get windowed positions")
        objects['xwin'], objects['ywin'] = get_winpos(
            objects['x'], objects['y'], objects['a'])
        if wcs is not None:
            objects['rawin'], objects['decwin'] = transform_method(
                objects['xwin'], objects['ywin'],0)
    
    # Calculate aperture flux
    if aper_radius is not None:
        objects['aper_radius'] = aper_radius
        flux, flux_err, flag = sep.sum_circle(
            img_data, objects['x'], objects['y'], aper_radius, gain=gain)
        objects['aper_flux'] = flux
        objects['aper_flux_err'] = flux_err
        objects['aper_flag'] = flag
        objects['aper_mag'] = -2.5*np.log10(objects['aper_flux']/exptime)
        if gain is not None:
            objects['aper_mag_err'] = 1.0857*np.sqrt(
                2*np.pi*aper_radius**2*bkg.globalrms**2+
                objects['aper_flux']/gain)/objects['aper_flux']
        else:
            objects['aper_mag_err'] = 1.0857*np.sqrt(
                2*np.pi*aper_radius**2*bkg.globalrms**2)/objects['aper_flux']
        # Get the pipeline flags for the image
        if dqmask_data is not None:
            objects['flags'] = get_img_flags(dqmask_data, 
                objects['x'], objects['y'],
                (2*aper_radius+1, 2*aper_radius+1),
                edge_val)
        # Calculate the positional error
        # See SExtractor Documentation for more on
        # ERRX2 and ERRY2
        # Ignore for now since this is very computationally
        # expensive with little to gain
        if False:
            back_data = bkg.back()
            err_x = np.zeros((len(objects,)), dtype=float)
            err_y = np.zeros((len(objects,)), dtype=float)
            err_xy = np.zeros((len(objects,)), dtype=float)
            for n, src in enumerate(objects):
                X = np.linspace(src['x']-aper_radius, src['x']+aper_radius, 
                    2*aper_radius+1)
                Y = np.linspace(src['y']-aper_radius, src['y']+aper_radius, 
                    2*aper_radius+1)
                X,Y = np.meshgrid(X,Y)
                flux = extract_array(img_data, 
                    (2*aper_radius+1, 2*aper_radius+1), (src['y'], src['x']))
                back = extract_array(back_data,
                    (2*aper_radius+1, 2*aper_radius+1), (src['y'], src['x']))
                if gain is not None and gain > 0:
                    sigma2 = back**2 + flux/gain
                else:
                    sigma2 = back**2
                flux2 = np.sum(flux)**2
                err_x[n] = np.sum(sigma2*(X-src['x'])**2)/flux2
                err_y[n] = np.sum(sigma2*(Y-src['y'])**2)/flux2
                err_xy[n] = np.sum(sigma2*(X-src['x'])*(Y-src['y']))/flux2
            objects['ERRX2'] = err_x
            objects['ERRY2'] = err_y
            objects['ERRXY'] = err_xy
        
    # include an index for each source that might be useful later on in
    # post processing
    objects['src_idx'] = [n for n in range(len(objects))]
    
    return objects, bkg

def get_winpos(data, x, y, a, subsampling=5):
    """
    Get windowed position. These are more accurate positions calculated
    by SEP (SExtractor) created by iteratively recentering on the area
    contained by the half-flux radius.
    
    Parameters
    ----------
    data: `~numpy.array`
        Image data
    x: array-like
        X coordinates of the sources
    y: array-like
        Y coordinates of the sources
    subsampling: int, optional
        Number of subpixels for each image pixel
        (used to calculate the half flux radius).
        Default=5.
    """
    import sep
    r, flag = sep.flux_radius(data, x, y, 
        6.*a, 0.5, subpix=subsampling)
    sig = 2. / 2.35 * r
    xwin, ywin, flags = sep.winpos(data, x, y, sig)
    return xwin, ywin