import logging
import astropy.units as apu

logger = logging.getLogger('astropyp.calibrate.phot')

def clean_sources(obs, mag_name, ref_name, check_columns=[], clipping=1):
    """
    Remove NaN values and clip sources outside a given number of standard deviations
    
    Parameters
    ----------
    obs: structured array-like
        astropy.table.QTable, pandas.DataFrame, or structured array of observations
    mag_name: str
        Name of the magnitude field
    ref_name: str
        Name of the reference catalog magnitude field
    check_columns: list of strings (optional)
        Names of columns to check for NaN values
    clipping: float (optional)
        Maximum number of standard deviations from the mean that a good source will be found.
        If clipping=0 then no standard deviation cut is made
    
    Returns
    -------
    good_sources: structure array-like
        Good sources from the original ``obs``.
    """
    import numpy as np
    
    # Remove NaN values for selected columns
    if len(check_columns)>0:
        conditions = [np.isfinite(obs[col]) for col in check_columns]
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond
        good_sources = obs[condition]
    else:
        good_sources = obs
    
    # Remove sources that have been flagged by SExtractor as bad
    good_sources = good_sources[(good_sources['FLAGS']==0) &
                                (good_sources['FLAGS_WEIGHT']==0)]
    
    # Remove the 5 brightest stars (might be saturated) and use range of 5 mags
    obs_min = np.sort(good_sources[mag_name])[5]
    obs_max = obs_min+5
    good_sources = (good_sources[(good_sources[mag_name]>obs_min) & 
        (good_sources[mag_name]<obs_max)])
    
    # Remove outliers
    diff = good_sources[mag_name]-good_sources[ref_name]
    good_sources = good_sources[np.sqrt((diff-np.mean(diff))**2)<np.std(diff)]
    
    return good_sources

def match_catalogs(cat1, cat2, ra1='XWIN_WORLD', dec1='YWIN_WORLD', 
        ra2='XWIN_WORLD', dec2='YWIN_WORLD', max_separation=1*apu.arcsec):
    """
    Use astropy.coordinates to match sources in two catalogs and 
    only select sources within a specified distance
    
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as apu
    
    if isinstance(max_separation, float) or isinstance(max_separation, int):
        max_separation = max_separation * apu.arcsec
    c1 = SkyCoord(cat1[ra1], cat1[dec1], unit='deg')
    c2 = SkyCoord(cat2[ra2], cat2[dec2], unit='deg')
    idx, d2, d3 = c1.match_to_catalog_sky(c2)
    matches = d2 < max_separation
    return idx, matches

def match_all_catalogs(catalogs, ra_names, dec_names, max_separation=1*apu.arcsec, min_detect=None):
    """
    Match a list of catalogs based on their ra, dec, and separation
    """
    if isinstance(ra_names, six.string_types):
        ra_names = [ra_names for n in range(len(catalogs))]
    if isinstance(dec_names, six.string_types):
        dec_names = [dec_names for n in range(len(catalogs))]
    catalog = catalogs[0]
    matches = np.array([True for n in range(len(catalog))])
    for n in range(1, len(catalogs)):
        idx, new_matches = match_catalogs(
            catalog, catalogs[n], ra_names[n-1], dec_names[n-1],
            ra_names[n], dec_names[n])
        matches = matches & new_matches
        catalogs[n] = catalogs[n][idx]
    for catalog in catalogs:
        catalog = catalog[matches]
    return catalogs

def combine_catalogs(expnums, catalog_names, idx_connect_str, dirpath, mag_name, 
        ra_names='XWIN_WORLD', dec_names='YWIN_WORLD', flux_names='FLUX_PSF', frames=None, 
        columns=None, max_separation=1*apu.arcsec, min_detect=None):
    """
    Combine a list of catalogs by matching sources and combining all frames
    into a single table
    """
    if isinstance(flux_names, six.string_types):
        flux_names = [flux_names for n in range(len(catalog_names))]
    # Set default columns
    if columns is None:
        columns = ['XWIN_WORLD', 'YWIN_WORLD', 'airmass', 'filename', 'frame', mag_name, 'MAG_PSF',
            'MAG_AUTO', 'MAGERR_PSF', 'MAGERR_AUTO', 'FLAGS', 'FLAGS_WEIGHT']
    # Get default frames
    if frames is None:
        hdulist = fits.open(os.path.join(dirpath, files.iloc[0].filename), memmap=True)
        frames = range(1,len(hdulist))
    
    # Match sources for each individual frame, then combine the frames
    all_frames = None
    for frame in frames:
        catalogs = []
        # Get sources in each catalog for the current frame
        for n,catalog_filename in enumerate(catalog_names):
            cat_frame = get_phot_params(expnums[n], catalog_filename, flux_names[n], 
                idx_connect_str, dirpath, mag_name, frame)
            catalogs.append(cat_frame[columns])
        # Only keep the sources found in at least min_detect catalogs
        catalogs = match_all_catalogs(catalogs, ra_names, dec_names, max_separation, min_detect)
        # Combine all of the frames for each catalog
        if all_frames is None:
            all_frames = catalogs
        else:
            all_frames = [vstack([all_frames[n], catalogs[n]]) for n in range(len(all_frames))]
    return all_frames

def calculate_magnitude(x, zero, color, extinct):
    """
    x[0] = reference in instrument band
    x[1] = reference in other band
    x[2] = airmass
    """
    #return (x[0]-zero+color*x[1]-extinct*x[2])/(1+color)
    return x[0] + zero + color*(x[0]-x[1]) + extinct*x[2]

def calibrate_standard(sources, mag_name, ref1_name, ref2_name, mag_err_name, ref1_err_name, 
        ref2_err_name, init_zero=-25, init_color=-.1, init_extinction=.1,
        fit_package='scipy'):
    """
    Calibrate a standard field with a set of refernce fields
    
    Parameters
    ----------
    sources: `astropy.table.QTable`
        Catalog of observations
    mag_name: str
        Name of the magniude column in ``sources``
    ref1_name: str
        Name of the reference column in ``sources`` in the same filter as ``mag_name``
    ref2_name: str
        Name of the reference column in ``sources`` to use for the color correction coefficient
    mag_err_name: str
        Name of the magnitude error column
    ref1_err_name: str
        Name of the error column for reference 1
    ref2_err_name: str
        Name of the error column for reference 2
    init_zero: float
        Initial guess for the zero point
    init_color: float:
        Initial guess for the color correction coefficient
    init_extinction: float
        Initial guess for the extinction coefficient
    """
    good_sources = sources
    init_params = [init_zero, init_color, init_extinction]
    instr_mag = good_sources[mag_name]
    ref_mag1 = good_sources[ref1_name]
    ref_mag2 = good_sources[ref2_name]
    airmass = good_sources['airmass']
    
    if fit_package=='scipy':
        from scipy.optimize import curve_fit
        x = [ref_mag1,ref_mag2,airmass]
        results = curve_fit(calculate_magnitude, x, instr_mag, init_params)
    elif fit_package=='statsmodels':
        import statsmodels.formula.api as smf
        good_sources['diff'] = good_sources[mag_name] - good_sources[ref1_name]
        good_sources['color'] = good_sources[ref1_name] - good_sources[ref2_name]
        result = smf.OLS.from_formula(formula='diff ~ color + airmass', data=good_sources).fit()
        results = [result.params.Intercept, result.params.color, result.params.airmass],result
    else:
        raise Exception("fit_package must be either 'statsmodels' or 'scipy'(default)")
    logger.debug("Zero point: {0}\nColor Correction: {1}\nExtinction: {2}\n".format(*results[0]))
    return results

def calibrate_2band(instr1, instr2, airmass1, airmass2, coeff1, coeff2):
    """
    This solves the set of equations:
        i_0 = i + A_i + C_i(i-z) + k_i X
        z_0 = z + A_z + C_z(z-i) + k_z X
    where i_0 and z_0 are the instrumental magnitudes, A_i and A_z are the zero points,
    C_i and C_z are the color terms, k_i and k_z are the atmospheric coefficients, 
    and X is the airmass.
    
    The solution is of the form:
        (1+C_i)i = b_i + C_i z
        (1+C_z)z = b_z + C_z i
    where
        b_i = i_0 - A_i - k_i X
        b_z = z_0 - A_z - k_z X
    so that
        i = (C_i b_z + C_z b_i + b_i) / d
        z = (C_z b_i + C_i b_z + b_z) / d
    where
        d = (1+C_i+C_z)
    
    Parameters
    ----------
    instr1: array-like
        Instrumental magnitudes of filter 1
    instr2: array-like
        Instrumental magnitudes of filter 2
    airmass1: array-like
        Airmass for each observation in filter 1
    airmass2: array-like
        Airmass for each observation in filter 2
    coeff1: array-like
        List of coeffients for calibrating instrumental magnitudes for instrument 1.
            * coeff1[0]: zeropoint
            * coeff1[1]: color coeffcient
            * coeff1[2]: extinction coefficient
    coeff2: array-like
        List of coeffients for calibrating instrumental magnitudes for instrument 2
    
    returns
    -------
    mag1: array-like
        Calibrated magnitude 1
    mag2: array-like
        Calibrated magnitude 2
    """
    b1 = instr1 - coeff1[0] - coeff1[2]*airmass1
    b2 = instr2 - coeff2[0] - coeff2[2]*airmass2
    d = 1 + coeff1[1] + coeff2[1]
    mag1 = (coeff1[1]*b2 + b1*(1+coeff2[1])) / d
    mag2 = (coeff2[1]*b1 + b2*(1+coeff1[1])) / d
    return (mag1,mag2)

def calibrate_1band(instr, airmass, coeff, color_band=None):
    """
    Given a solution for z from calibrate_iz, this returns a Y magnitude using:
        Y_0 = Y + A_Y + C_Y(Y-z) + k_Y X
    where Y0 is the instrumental magnitude, A_Y is the zero point, C_Y is the color coefficent, 
    k_Y is the extinction coefficient, and X is the airmass
    """
    if color_band is not None:
        mag = (instr - coeff[0] + coeff[1]*color_band - coeff[2]*airmass)/(1+coeff[1])
    else:
        mag = instr - coeff[0] - coeff[1]*airmass
    return mag

def get_phot_params(expnum, catalog_filename, flux_name, idx_connect_str, dirpath, mag_name, frame):
    """
    Get necessary parameters like airmass and exposure time from the FITS image headers
    """
    from astropyp import index
    from astropy.io import fits
    import numpy as np
    from astropy.table import QTable
    
    sql = "select * from decam_files where EXPNUM={0} and PRODTYPE='image'".format(expnum)
    files = index.query(sql, idx_connect_str)
    
    # Get fields from FITS header
    hdulist = fits.open(os.path.join(dirpath, files.iloc[0].filename), memmap=True)
    header = hdulist[0].header
    exptime = header['exptime']
    airmass = 1/np.cos(np.radians(header['ZD']))
    header = hdulist[frame].header
    gain = header['arawgain']
    logger.info('{0}: airmass={1}, exptime={2}'.format('', airmass, exptime))
    
    # add fields to catalog
    catalog = QTable.read(catalog_filename, hdu=frame)
    catalog[flux_name][catalog[flux_name] * gain/exptime ==0] = np.nan
    catalog[mag_name] = -2.5*np.log10(catalog[flux_name] * gain/exptime)
    catalog['airmass'] = airmass
    catalog['filename'] = catalog_filename
    catalog['frame'] = frame
    return catalog