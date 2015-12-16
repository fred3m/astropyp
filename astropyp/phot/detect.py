import numpy as np
from astropy import table
from astropy.nddata.utils import extract_array

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
        wcs=None, aper_radius=None, windowed=True, edge_val=1):
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
    """
    import sep
    
    # Estimate the background and subtract it (in place) from the image
    # Note: we might not need to do background subtraction due to the 
    # community pipeline
    bkg = sep.Background(img_data, dqmask_data)
    if subtract_bkg:
        bkg.subfrom(img_data)
    
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
    if 'ra' not in objects.columns.keys() and wcs is not None:
        objects['ra'], objects['dec'] = wcs.all_pix2world(objects['x'], objects['y'], 0)
    if 'x' not in objects.columns.keys():
        if wcs is None:
            raise Exception("You must provide a wcs transformation if specifying ra and dec")
        objects['x'], objects['y'] = wcs.all_world2pix(objects['ra'], objects['dec'], 0)
    
    if windowed:
        # Calculate the kron radius
        if 'kron_radius' not in objects:
            objects['kron_radius'], flags = sep.kron_radius(
                img_data, objects['x'], objects['y'], objects['a'], 
                objects['b'], objects['theta'], 6.0)
            objects['kron_flag'] = flags
        eff_radius = objects['kron_radius']*np.sqrt(objects['a']*objects['b'])
        use_circle = (sex_params['kron_k']*eff_radius < 
                      sex_params['kron_min_radius'])
        # Calculate the flux in the kron_radius
        objects['kron_flux'] = np.nan
        objects['kron_flux_err'] = np.nan
        objects['kron_flag'] = flags
        kron_flux, kron_err, flags = sep.sum_ellipse(
            img_data, objects['x'][~use_circle], objects['y'][~use_circle], 
            objects['a'][~use_circle], objects['b'][~use_circle], 
            objects['theta'][~use_circle], 
            sex_params['kron_k']*objects['kron_radius'][~use_circle], 
            subpix=1, gain=gain)
        objects['kron_flux'][~use_circle] = kron_flux
        objects['kron_flux_err'][~use_circle] = kron_err
        objects['kron_flag'][~use_circle]=(
            objects['kron_flag'][~use_circle] | flags)
        
        objects['kron_eff_radius'] = eff_radius
        # If the kron radius is too small, use the minimum circular aperture radius
        flux, flux_err, flags = sep.sum_circle(
            img_data, objects['x'][use_circle], objects['y'][use_circle], 
            sex_params['kron_min_radius'], 
            subpix=1, gain=gain)
        objects['kron_flux'][use_circle] = flux
        objects['kron_flux_err'][use_circle] = flux_err
        objects['kron_flag'][use_circle]=objects['kron_flag'][use_circle]|flags

        # Calculate the flux radius for half of the total flux (needed to get windowed positions)
        objects['half_flux_radius'], flags = sep.flux_radius(img_data, 
            objects['x'], objects['y'], 6.*objects['a'], 0.5, 
            normflux=objects['kron_flux'], subpix=5)

        # Get the windowed positions
        sig = 2./2.35 * objects['half_flux_radius']
        objects['xwin'], objects['ywin'], flags = sep.winpos(img_data, 
            objects['x'], objects['y'], sig)
    
        # Set windowed WCS if possible
        if wcs is not None:
            objects['ra_win'], objects['dec_win'] = wcs.all_pix2world(
                objects['xwin'], objects['ywin'], 0)
        if exptime is not None:
            # Calculate kron magnitudes
            objects['kron_mag'] = -2.5*np.log10(objects['kron_flux']/exptime)
            if gain is not None:
                objects['kron_mag_err'] = 1.0857*np.sqrt(
                    2*np.pi*objects['kron_radius']**2*bkg.globalrms**2+
                    objects['kron_flux']/gain)/objects['kron_flux']
            else:
                objects['kron_mag_err'] = 1.0857*np.sqrt(
                    2*np.pi*objects['kron_radius']**2*bkg.globalrms**2)
    
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
    # include an index for each source that might be useful later on in
    # post processing
    objects['src_idx'] = [n for n in range(len(objects))]
    
    return objects, bkg