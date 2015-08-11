import logging

logger = logging.getLogger('astropyp.calibrate.query')

catalog_info = {
    'SDSS9': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'e_RAJ2000',
            'e_dec': 'e_DEJ2000',
            'pm_ra': 'pmRA',
            'pm_dec': 'pmDE',
            'e_pm_ra': 'e_pmRA',
            'e_pm_dec': 'e_pmDE',
            'ObsDate': 'ObsDate'
        },
        'info': {
            'jyear': 'ObsDate',
            'pm_units': 'mas',
            'e_pos_units': 'arcsec',
            'vizier_id': "V/139"
        }
    },
    'UKIDSS9': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'e_RAJ2000',
            'e_dec': 'e_DEJ2000',
            'pm_ra': 'pmRA',
            'pm_dec': 'pmDE',
            'e_pm_ra': 'e_pmRA',
            'e_pm_dec': 'e_pmDE',
            'Epoch': 'Epoch'
        },
        'info': {
            'jyear': 'Epoch',
            'pm_units': 'mas',
            'e_pos_units': 'mas',
            'vizier_id': "II/319"
        }
    },
    '2MASS': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'errMaj',
            'e_dec': 'errMin',
            'e_PA': 'errPA',
        },
        'info': {
            'epoch': 2000.0,
            'e_pos_units': 'arcsec',
            'vizier_id': 'II/246'
        }
    },
    'AllWISE': {
        'columns': {
            'ra': 'RA_pm', # position at MJD=55400.0 (2010.5589)
            'dec': 'DE_pm',
            'e_ra': 'e_RA_pm',
            'e_dec': 'e_DE_pm',
            'pm_ra': 'pmRA',
            'pm_dec': 'pmDE',
            'e_pm_ra': 'e_pmRA',
            'e_pm_dec': 'e_pmDE',
        },
        'info': {
            'epoch': 2010.5589,
            'pm_units': 'mas',
            'e_pos_units': 'arcsec',
            'vizier_id': 'II/328'
        }
    },
    'UCAC4': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'ePos',
            'e_dec': 'ePos',
            'pm_ra': 'pmRA',
            'pm_dec': 'pmDE',
            'e_pm_ra': 'e_pmRA',
            'e_pm_dec': 'e_pmDE',
        },
        'info': {
            'epoch': 2000.0,
            'pm_units': 'mas',
            'e_pos_units': 'mas',
            'vizier_id': 'I/322A'
        }
    },
    'GSC': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'e_RAdeg',
            'e_dec': 'e_DEdeg',
            'epoch': 'Epoch'
        },
        'info': {
            'jyear': 'Epoch',
            'e_pos_units': 'arcsec',
            'vizier_id': 'I/305'
        }
    },
    'DENIS': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'epoch': 'ObsJD'
        },
        'info': {
            'mjd': 'ObsJD',
            'e_pos_units': 'mas',
            'e_pos': '400',
            'vizier_id': 'B/denis'
        }
    },
    'USNOB1': {
        'columns': {
            'ra': '_RAJ2000',
            'dec': '_DEJ2000',
            'e_ra': 'e_RAJ2000',
            'e_dec': 'e_DEJ2000',
            'pm_ra': 'pmRA',
            'pm_dec': 'pmDE',
            'e_pm_ra': 'e_pmRA',
            'e_pm_dec': 'e_pmDE',
            'epoch': 'Epoch'
        },
        'info': {
            'jyear': 'Epoch',
            'pm_units': 'mas',
            'e_pos_units': 'mas',
            'vizier_id': 'I/284'
        }
    }
}

def get_query_region(ra_bounds, dec_bounds, print_results=True):
    """
    Get the max/min ra and dec based on lists of boundaries
    
    Parameters
    ----------
    ra_bounds: list
        Boundary values for RA. These are usually taken from the header, for example
        ra_bounds = [hdu.header['CORN1RA'], hdu.header['CORN2RA'], hdu.header['CORN3RA'], 
        hdu.header['CORN4RA']]
    dec_bounds: list
        Boundary values for DEC.
    
    Returns
    -------
    result: tuple
        The result is the tuple (ra_min, ra_max, dec_min, dec_max)
    """
    min_ra = min(ra_bounds)
    max_ra = max(ra_bounds)
    min_dec = min(dec_bounds)
    max_dec = max(dec_bounds)
    if print_results:
        logger.debug('ra range: {0} to {1}'.format(min_ra, max_ra))
        logger.debug('dec range: {0} to {1}'.format(min_dec, max_dec))
    return min_ra, max_ra, min_dec, max_dec

def query_cat(catalog, min_ra, max_ra, min_dec, max_dec, columns=None,
        column_filters=None):
    """
    Use vizquery to get a reference catalog from vizier
    """
    from astroquery.vizier import Vizier
    import numpy as np
    from astropy.coordinates import SkyCoord
    # Build vizquery statement
    width = int(np.ceil((max_ra-min_ra)*60))
    height = int(np.ceil((max_dec-min_dec)*60))
    center = SkyCoord((min_ra+max_ra)/2, (min_dec+max_dec)/2, unit='deg')
    
    # If no column filters are specified, use the defaults
    if column_filters is None:
        if catalog.startswith('SDSS'):
            column_filters = {
                'cl': '=6',
                'q_mode':'=+'
            }
        elif catalog.startswith('UKIDSS'):
            column_filters = {
                'cl': '=-1',
                'm': '=1'
            }
        else:
            column_filters = {}
    # Query the catalog in Vizier
    logger.debug('columns:{0}'.format(columns))
    v = Vizier(columns=columns, column_filters=column_filters, 
        catalog=catalog_info[catalog]['info']['vizier_id'])
    v.ROW_LIMIT=200000
    result = v.query_region(center, width='{0}m'.format(width*1.25), 
        height='{0}m'.format(height*1.25))
    refcat = result[0]
    
    return refcat

def update_refcat(cat_name, refcat, obs_dates):
    """
    Update a reference catalogs errors, since the positional errors in Vizier are
    for the J2000 positions before the proper motions are used to move the positions
    to the observed epoch.
    """
    import numpy as np
    import astropy.units as apu
    
    cat_info = catalog_info[cat_name]
    
    # set the epoch of the reference observation
    if 'epoch' in cat_info['info']:
        refcat['epoch'] = cat_info['info']['epoch']
    elif 'jyear' in cat_info['info']:
        refcat.rename_column(cat_info['info']['jyear'], 'epoch')
    elif 'jd' in cat_info['info']:
        from astropy.time import Time
        jd = Time(refcat[cat_info['info']['jd']], format='jd')
        refcat['epoch'] = jd.jyear
        del refcat[cat_info['info']['jd']]
    
    # Change the column names to fit a standard
    for colname in cat_info['columns']:
        if ((colname=='e_ra' or colname=='e_dec') and 
                cat_info['columns']['e_ra']==cat_info['columns']['e_dec']):
            refcat[colname] = refcat[cat_info['columns'][colname]].astype(float)
            refcat[colname].unit = refcat[cat_info['columns'][colname]].unit
        elif cat_info['columns'][colname] in refcat.columns:
            x = refcat[cat_info['columns'][colname]]
            refcat[colname] = x.astype(float)
            refcat[colname].unit = x.unit
            # Astropy uses masks instead of nan values, so we convert the
            # mask to NaN
            refcat[colname][x.mask] = np.nan
            del refcat[cat_info['columns'][colname]]
            #refcat.rename_column(cat_info['columns'][colname], colname)
    # Change proper motion errors and position errors to mas
    for col in ['pm_ra', 'pm_dec', 'e_pm_ra', 'e_pm_dec']:
        if col in cat_info['columns']:
            refcat[col].convert_unit_to(apu.mas/apu.year)
    if 'e_ra' in refcat.columns and 'e_dec' in refcat.columns:
        refcat['e_ra'].convert_unit_to(apu.mas)
        refcat['e_dec'].convert_unit_to(apu.mas)
    else:
        warnings.warn("{0} was missing 'e_ra' and 'e_dec'".format(cat_name))
    
    # Tables do not add,subtract, multiply, or divide quantities properly so
    # we need to incldue a conversion factor from mas to deg
    mas2deg = 1/3600000.0
    # Update positions to the observation dates
    if 'pm_ra' in refcat.columns:
        for obs_date in obs_dates:
            date_diff = obs_date-refcat['epoch']
            # Calculate the new positions
            ra_field = 'ra_J{0:.1f}'.format(obs_date)
            dec_field = 'dec_J{0:.1f}'.format(obs_date)
            delta_ra = date_diff*refcat['pm_ra']*mas2deg
            delta_ra.unit = 'deg'
            delta_dec = date_diff*refcat['pm_dec']*mas2deg
            delta_dec.unit = 'deg'
            refcat[ra_field] = refcat['ra']+delta_ra
            refcat[dec_field] = refcat['dec']+delta_dec
            refcat[ra_field].unit='deg'
            refcat[dec_field].unit='deg'
            # Calculate the errors in the new positions and convert to mas
            refcat['e_'+ra_field] = np.sqrt(
                refcat['e_ra']**2+(date_diff*refcat['e_pm_ra'])**2)
            refcat['e_'+dec_field] = np.sqrt(
                refcat['e_dec']**2+(date_diff*refcat['e_pm_dec'])**2)
            refcat['e_'+ra_field].unit = 'mas'
            refcat['e_'+dec_field].unit = 'mas'
            # Some sources might have NaN values for pm
            pm_isnan = (np.isnan(refcat['pm_ra']) | np.isnan(refcat['pm_dec']) | 
                (refcat['pm_ra'].mask) | (refcat['pm_dec'].mask))
            refcat[ra_field][pm_isnan] = refcat['ra'][pm_isnan]
            refcat[dec_field][pm_isnan] = refcat['dec'][pm_isnan]
            refcat['e_'+ra_field][pm_isnan] = refcat['e_ra'][pm_isnan]
            refcat['e_'+dec_field][pm_isnan] = refcat['e_dec'][pm_isnan]
    
    return refcat

def cds_query(pipeline, obj, catalog, columns=None, frames=None,
        proctype='InstCal', filter_columns=None, obs_dates=None):
    """
    Query vizier and return a reference catalog
    
    Parameters
    ----------
    pipeline: astromatic_wrapper.Pipeline
        Pipeline containing index info
    obj: str
        Name of the DECam object
    catalog: str
        Name of the reference catalog. This can either be a name, like ``SDSS9``, or a 
        Vizier Id, like ``V/139``
    columns: str (optional)
        Space separated list of columns to query. If the first character is a ``*`` Vizier
        will return all of the default columns
    frames: list (optional)
        Sometimes a FOV might have too many matches for a single query, so the query can be done
        frame by frame. If ``frames=[]`` then each frame in the fits file will be queried,
        otherwise only the frames in the list will be queried.
    proctype: str (optional)
        DECam pipeline proctype. Default is ``InstCal``
    filter_columns: dict
        Parameters to pass to astroquery. The keys are the names of a column in the
        reference catalog and the values are an operator (such as '>','<','=') and
        a value. For example ``filter_columns={'e_pmRA':'<200'}.
    """
    from astropyp import index
    from astorpy.table import vstack
    from astropy.time import Time
    
    # Load a fits image for the given object
    sql = "select * from decam_obs where object like '{0}%'".format(obj)
    exposures = index.query(sql, pipeline.idx_connect_str).sort(['expnum'])
    exp = exposures.iloc[0]
    sql="select * from decam_files where expnum={0} and proctype='{1}'".format(
        exp['expnum'], proctype
    )
    sql += " and prodtype='image'"
    files = index.query(sql, pipeline.idx_connect_str).iloc[0]
    hdulist = fits.open(files['filename'], memmap=True)
    if obs_dates is None:
        dates = exposures['cal_date'].unique().tolist()
        obs_dates = Time(dates, format='iso').jyear
    logger.info('Loading {0} for {1}'.format(catalog, obj))
    
    if frames is None:
        # Get the ra and dec range for the entire field of view
        header = hdulist[0].header
        min_ra, max_ra, min_dec, max_dec = get_query_region(
            [header['CORN1RA'], header['CORN2RA'], header['CORN3RA'], header['CORN4RA']],
            [header['CORN1DEC'], header['CORN2DEC'], header['CORN3DEC'], header['CORN4DEC']])
        # Query vizier
        refcat = query_cat(catalog, min_ra, max_ra, min_dec, max_dec, columns, filter_columns)
    else:
        # For each frame, query Vizier and merge the result into a catalog for the entire field
        if len(frames)==0:
            frames = range(1, len(hdulist))
        refcat = None
        for frame in frames:
            logger.info('querying frame: {0}'.format(frame))
            # Get the ra and dec range for the current frame
            header = hdulist[frame].header
            min_ra, max_ra, min_dec, max_dec = get_query_region(
                [header['COR1RA1'], header['COR2RA1'], header['COR3RA1'], header['COR4RA1']],
                [header['COR1DEC1'], header['COR2DEC1'], header['COR3DEC1'], header['COR4DEC1']])
            # Query vizier
            new_ref = query_cat(catalog, min_ra, max_ra, min_dec, max_dec, columns, filter_columns)
            logger.debug('new entries: {0}'.format(len(new_ref)))
            # Merge the results
            if refcat is None:
                refcat = new_ref
            else:
                refcat = vstack([refcat, new_ref])
                logger.debug('total entries: {0}'.format(len(refcat)))
    # Update the position errors and other fields needed for better SCAMP astrometric fit
    refcat = update_refcat(catalog, refcat, obs_dates)
    # Convert the Table to a fits_ldac file to read into SCAMP
    # Sometimes the meta data is too long to save to a fits file, in which case
    # we just delete the meta data
    try:
        import astromatic_wrapper as aw
        new_hdulist = aw.utils.ldac.convert_table_to_ldac(refcat)
    except ValueError:
        refcat.meta={}
        new_hdulist = aw.utils.ldac.convert_table_to_ldac(refcat)
    cat_path = os.path.join(pipeline.paths['catalogs'], 'ref', "{0}-{1}.fits".format(obj, catalog))
    new_hdulist.writeto(cat_path, clobber=True)
    logger.info('saved {0}'.format(cat_path))