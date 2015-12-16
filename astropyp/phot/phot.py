import logging
import astropy.units as apu
try:
    import six
except ImportError:
    try:
        from astropy.extern import six
    except:
        raise Exception("Unable to import six module")

logger = logging.getLogger('astropyp.calibrate.phot')

def old_clean_sources(obs, mag_name, ref_name, check_columns=[], clipping=1):
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

def clean_sources(obs, mag_names, ref_name, check_columns=[], err_columns=[], 
        clipping=None, brightness_limit=True):
    """
    Remove NaN values from a list of check_columns and clip based on the values of
    certain error columns
    
    Parameters
    ----------
    obs: structured array-like
        astropy.table.Table, pandas.DataFrame, or structured array of observations
    mag_names: str or list
        Name of the magnitude field
    check_columns: list of strings (optional)
        Names of columns to check for NaN values
    err_columns: list (optional)
        List of columns to check for errors
    clipping: float or list of floats (optional)
        Maximum error in err_columns to include in clean sources. If a list is
        given then each error column is checked against its corresponding entry
        in the clipping list.
    
    Returns
    -------
    good_sources: structure array-like
        Good sources from the original ``obs``.
    """
    import numpy as np
    from astropy.extern import six
    
    # Check if mag_names is a list or string
    if isinstance(mag_names, six.string_types):
        mag_names = [mag_names]
    # Remove NaN values for selected columns
    if len(check_columns)>0:
        conditions = [np.isfinite(obs[col]) for col in check_columns]
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond
        good_sources = obs[condition]
    else:
        good_sources = obs
    
    def temp_check(sources, checkpoint):
        ass = sources[(sources['NUMBER']==666)&(sources['ccd']==59)]
        if len(ass)>0:
            logger.info('made it past {0}'.format(checkpoint))
    temp_check(good_sources, "check_columns")
    # Mask NaN and infinite values
    if not good_sources.masked:
        from astropy.table import Table
        good_sources = Table(good_sources, masked=True)
    for col in good_sources.columns.keys():
        # Strings cannot be checked by isfinite, so we ignore the type error they throw
        try:
            good_sources[col].mask = ~np.isfinite(good_sources[col])
        except TypeError:
            pass
    temp_check(good_sources, "mask")
    # Remove the 5 brightest stars (might be saturated) and use range of 5 mags
    if brightness_limit:
        for mag_name in mag_names:
            obs_min = np.sort(good_sources[mag_name])[5]
            obs_max = obs_min+5
            good_sources = (good_sources[(good_sources[mag_name]>obs_min) & 
                (good_sources[mag_name]<obs_max)])
    temp_check(good_sources, "brightness")
    # Based on error columns
    if len(err_columns)>0:
        if clipping is None:
            raise Exception("You did not specify a value or list of values to use for clipping")
        if isinstance(clipping, float) or isinstance(clipping, int):
            clipping = [clipping for n in range(len(err_columns))]
        for n,col in enumerate(err_columns):
            good_sources = good_sources[good_sources[col]<clipping[n]]
    temp_check(good_sources, "error_columns")
    # Some sources have bad PSF fits so we check the the PSF 
    # magnitudes and Aperture Magnitudes are nearly the same
    good_sources = good_sources[np.abs(good_sources['MAG_AUTO']-good_sources['MAG_PSF'])<.02]
    # Remove sources that have been flagged by SExtractor as bad
    #if 'FLAGS' in good_sources.columns:
    #    good_sources = good_sources[good_sources['FLAGS']==0]
    #if 'FLAGS_WEIGHT' in good_sources.columns:
    #    good_sources = good_sources[good_sources['FLAGS_WEIGHT']==0]
    temp_check(good_sources, "all")

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

def match_all_catalogs(catalogs, ra_names, dec_names, max_separation=1*apu.arcsec, 
        min_detect=None, combine=True):
    """
    Match a list of catalogs based on their ra, dec, and separation
    """
    import numpy as np
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
    for n in range(len(catalogs)):
        catalogs[n] = catalogs[n][matches]
    if combine:
        from astropy.table import vstack
        catalogs = vstack(catalogs)
    return catalogs

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
        fit_package='scipy', airmass_name='airmass'):
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
    airmass = good_sources[airmass_name]
    
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

def calibrate_2band(instr1, instr2, airmass1, airmass2, coeff1, coeff2,
        zero_key='zero', color_key='color', extinct_key='extinction'):
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
    b1 = instr1 - coeff1[zero_key] - coeff1[extinct_key]*airmass1
    b2 = instr2 - coeff2[zero_key] - coeff2[extinct_key]*airmass2
    d = 1 + coeff1[color_key] + coeff2[color_key]
    mag1 = (coeff1[color_key]*b2 + b1*(1+coeff2[color_key])) / d
    mag2 = (coeff2[color_key]*b1 + b2*(1+coeff1[color_key])) / d
    return (mag1,mag2)

def calibrate_1band(instr, airmass, coeff, color_band=None, zero_key='zero',
        color_key='color', extinct_key='extinction'):
    """
    Given a solution for z from calibrate_iz, this returns a Y magnitude using:
        Y_0 = Y + A_Y + C_Y(Y-z) + k_Y X
    where Y0 is the instrumental magnitude, A_Y is the zero point, C_Y is the color coefficent, 
    k_Y is the extinction coefficient, and X is the airmass
    """
    if color_band is not None:
        mag = (instr - coeff[zero_key] + coeff[color_key]*color_band - 
            coeff[extinct_key]*airmass)/(1+coeff[color_key])
    else:
        mag = instr - coeff[zero_key] - coeff[color_key]*airmass
    return mag