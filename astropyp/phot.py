import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger('decamtoyz.phot')

def build_rough_coeff_plot(sources, mag_name, ref_name):
    """
    Sometimes it is helpful to plot the average diffence for a given FOV or CCD.
    This can help identify observations that were not photometric as the
    avg difference vs airmass should be roughly linear.
    
    Parameters
    ----------
    sources: `astropy.table.Table`
        Catalog of observations
    mag_name: str
        Name of the magnitude column in ``sources``
    ref_name: str
        Name of the reference magnitude column in ``sources
    """
    x = []
    y = []
    err = []
    for airmass in np.unique(sources['airmass']).tolist():
        x.append(airmass)
        obs = sources[sources['airmass']==airmass]
        obs = obs[np.isfinite(obs[mag_name]) & np.isfinite(obs[ref_name])]
        diff = obs[mag_name]-obs[ref_name]
        obs = obs[np.sqrt((diff-np.mean(diff))**2)<np.std(diff)]
        median = np.median(obs[mag_name]-obs[ref_name])
        y.append(median)
        err.append(np.std(obs[mag_name]-obs[ref_name]))
    plt.xlabel('Airmass')
    plt.ylabel('Median difference in {0} magnitude'.format(mag_name))
    plt.plot(x,y)
    plt.errorbar(x, y, yerr=err, fmt='-o')
    plt.show()

def build_diff_plot(obs, mag_name, ref_name, mag_err_name, ref_err_name, 
        plot_format='b.', show_stats=True, clipping=1, filename=None):
    """
    Plot the difference between reference magnitudes and observed magnitudes for a 
    given set of observations
    
    Parameters
    ----------
    obs: `astropy.table.Table`
        Catalog of observations to compare
    mag_name: str
        Name of the magniude column in ``obs`` to compare
    ref_name: str
        Name of the reference column in ``obs`` to compare
    mag_err_name: str
        Name of the magnitude error column
    ref_err_name: str
        Name of the reference error column
    plot_format: str
        Format for matplotlib plot points
    show_stats: str
        Whether or not to show the mean and standard deviation of the observations
    """
    diff = obs[mag_name]-obs[ref_name]
    # Remove outlier sources
    plot_obs = obs[np.sqrt((diff-np.mean(diff))**2) < clipping*np.std(diff)]
    # build plot
    x = plot_obs[mag_name]
    y = plot_obs[mag_name]-plot_obs[ref_name]
    err = np.sqrt(plot_obs[mag_err_name]**2+plot_obs[ref_err_name]**2)
    plt.errorbar(x, y, yerr=err, fmt=plot_format)
    plt.xlabel(mag_name)
    plt.ylabel('Diff from Std Sources')
    # show stats
    if show_stats:
        logger.info('mean: {0}'.format(np.mean(y)))
        logger.info('std dev: {0}'.format(np.std(y)))
    # Save plot or plot to screen
    if filename is None:
        plt.show()
    else:
        plt.save(filename)
    plt.close()
    return y

def clean_sources(obs, mag_name, ref_name, check_columns=[], clipping=1):
    """
    Remove NaN values and clip sources outside a given number of standard deviations
    
    Parameters
    ----------
    obs: structured array-like
        astropy.table.Table, pandas.DataFrame, or structured array of observations
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
    sources: `astropy.table.Table`
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

def calibrate_nonlinear_standard(sources, mag_name, ref1_name, ref2_name, 
        mag_err_name, ref1_err_name, ref2_err_name, init_zero=-25, init_color=-.1, 
        init_extinction=.1, fit_package='scipy'):
    """
    Calibrate a standard field with a set of refernce fields
    
    Parameters
    ----------
    sources: `astropy.table.Table`
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
        good_sources['color2'] = good_sources['color']**2
        good_sources['color_airmass'] = good_sources['color']*good_sources['airmass']
        result = smf.OLS.from_formula(formula='diff ~ color + color2 + + color_airmass + airmass', 
            data=good_sources).fit()
        results = [result.params.Intercept, result.params.color, result.params.color2,
            result.params.airmass, result.params.color_airmass],result
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
    from decamtoyz.index import query_idx
    sql = "select * from decam_files where EXPNUM={0} and PRODTYPE='image'".format(expnum)
    files = query_idx(sql, idx_connect_str)
    
    # Get fields from FITS header
    hdulist = fits.open(os.path.join(dirpath, files.iloc[0].filename), memmap=True)
    header = hdulist[0].header
    exptime = header['exptime']
    airmass = 1/np.cos(np.radians(header['ZD']))
    header = hdulist[frame].header
    gain = header['arawgain']
    logger.info('{0}: airmass={1}, exptime={2}'.format('', airmass, exptime))
    
    # add fields to catalog
    catalog = Table.read(catalog_filename, hdu=frame)
    catalog[flux_name][catalog[flux_name] * gain/exptime ==0] = np.nan
    catalog[mag_name] = -2.5*np.log10(catalog[flux_name] * gain/exptime)
    catalog['airmass'] = airmass
    catalog['filename'] = catalog_filename
    catalog['frame'] = frame
    return catalog