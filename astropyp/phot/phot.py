import logging
from collections import OrderedDict
import numpy as np
import astropy.units as apu

import astropyp.catalog
from astropyp.phot import detect

try:
    import six
except ImportError:
    try:
        from astropy.extern import six
    except:
        raise Exception("Unable to import six module")

logger = logging.getLogger('astropyp.calibrate.phot')

class SingleImage:
    """
    Collection of Groups and PSFs for an entire image
    
    Parameters
    ----------
    catalog: `~astropyp.catalog.ImageCatalog` or `~astropy.table.Table`
        Catalog of sources with mappings to standard catalog fields.
        See `~astropyp.catalog.ImageCatalog` for more on ImageCatalogs.
        If an `~astropy.table.Table` is passed instead it will be
        converted to an ImageCatalog with the standard mapping
        to x,y,ra,dec,etc.
    """
    def __init__(self, header=None, img=None, dqmask=None, wtmap=None,
            wcs=None, separation=None, gain=None, exposure=None, 
            exptime=None, catalog=None, psf=None,
            subsampling=5, aper_radius=None, bkg=None,
            cluster_method='dbscan', mask_group=True,
            show_plots=False, groups=[], indices=OrderedDict()):
        
        self.header = header
        self.img = img
        self.dqmask = dqmask
        self.wtmap = wtmap
        self.wcs = wcs
        self.separation = separation
        self.gain = gain
        self.catalog = catalog
        self.exposure = exposure
        self.exptime = exptime
        self.psf = psf
        self.subsampling = subsampling
        self.aper_radius = aper_radius
        self.bkg = bkg
        self.cluster_method = cluster_method
        self.mask_group = mask_group
        self.show_plots = show_plots
        self.groups = groups
        self.indices = indices
        
        if not isinstance(self.catalog, astropyp.catalog.ImageCatalog):
            self.catalog = astropyp.catalog.ImageCatalog(self.catalog)
        # If the image is part of an focal array from a larger exposure,
        # use the exposure settings to set parameters
        if self.exptime is None and self.exposure is not None:
            self.exptime = exposure.exptime
        
        self.group_ids = range(len(self.groups))
    
    def detect_sources(self, sex_params={}, aper_radius=None, 
            subtract_bkg=False, gain=None, wcs=None, exptime=None,
            windowed=True, edge_val=1):
        # Set optional parameters
        if aper_radius is None:
            aper_radius = self.aper_radius
        elif self.aper_radius is None:
            self.aper_radius = aper_radius
        if gain is None:
            gain = self.gain
        elif self.gain is None:
            self.gain = gain
        if wcs is None:
            wcs = self.wcs
        elif self.wcs is None:
            self.wcs = wcs
        if exptime is None:
            exptime = self.exptime
        elif self.exptime is None:
            self.exptime = exptime
        
        result = detect.get_sources(self.img, self.dqmask, self.wtmap, 
            exptime, sex_params, None, subtract_bkg, gain, 
            wcs, aper_radius, windowed, edge_val)
        sources, self.bkg = result
        self.catalog = astropyp.catalog.ImageCatalog(
            sources, a='a', b='b', peak='peak')
        
        if hasattr(self.exposure, 'airmass'):
            self.catalog.sources['airmass'] = self.exposure.airmass
            self.catalog.update_static_column('airmass', 'airmass')
        return self.catalog, self.bkg
    
    def select_psf_sources(self,
            min_flux=None, min_amplitude=None, min_dist=None, max_ratio=None, 
            edge_dist=None, verbose=True, aper_radius=None,
            units='deg', badpix_flags=['flags'], flag_max=0, psf_idx='psf'):
        """
        Select sources with a minimum flux, amplitude, separation,
        and circular shape to use to build the psf.
        """
        if aper_radius is None:
            aper_radius = self.aper_radius
        elif self.aper_radius is None:
            self.aper_radius = aper_radius
        
        result = astropyp.phot.psf.select_psf_sources(self.img, 
            self.catalog, aper_radius, min_flux, min_amplitude, min_dist, 
            max_ratio, edge_dist, verbose, 
            badpix_flags=badpix_flags, flag_max=flag_max)
        self.indices[psf_idx], self.flags = result
        return result
    
    def create_psf(self, psf_sources=None, psf_radius=None, 
            combine_mode='median', offset_buffer=3):
        if psf_sources is None:
            psf_sources = 'psf'
        if psf_radius is None:
            psf_radius = self.aper_radius
        
        if isinstance(psf_sources, six.string_types):
            psf_sources = self.catalog.sources[self.indices[psf_sources]]
        
        psf_array = astropyp.phot.psf.build_psf(
            self.img, psf_radius, psf_sources, 
            subsampling=self.subsampling, combine_mode=combine_mode, 
            offset_buffer=offset_buffer)
        self.psf = astropyp.phot.psf.SinglePSF(psf_array, 1., 0, 0, 
            self.subsampling)
        return psf_array
    
    def show_psf(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D

        X = np.arange(0, self.psf._psf_array.shape[1], 1)
        Y = np.arange(0, self.psf._psf_array.shape[0], 1)
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure(figsize=(10, 10))
        ax=fig.add_subplot(1,1,1, projection='3d')
        ax.plot_wireframe(X, Y, self.psf._psf_array)#, rstride=5, cstride=5)
        plt.show()
    
    def create_psf_groups(self, separation=None, cluster_method='dbscan',
            verbose=False):
        """
        Group sources with overlapping PSF's
        """
        if separation is None:
            separation = self.psf._width
        
        if cluster_method=='dbscan':
            # If user has sklearn installed, use DBSCAN to cluster the objects
            # in groups with overlapping PSF's
            try:
                from sklearn.cluster import DBSCAN
            except ImportError:
                Exception(
                    "You must install sklearn to use 'dbscan' clustering")
            
            positions = np.array(zip(self.catalog.x, self.catalog.y))
            # Compute DBSCAN
            db = DBSCAN(eps=separation, min_samples=1).fit(positions)
            self.db = db
            self.groups = OrderedDict()
            self.group_ids = np.unique(db.labels_)
            group_indices = db.labels_
            self.indices['group'] = group_indices
        else:
            raise Exception(
                "cluster_method {0} is not currently supported".format(
                    cluster_method))
            
        # If a 'peak' field has not been defined in the catalog,
        # use the pixel value at each points position as the 
        # initial amplitude for the fit
        amplitudes = self.catalog.peak
        if amplitudes is None:
            amplitudes = self.img[
                self.catalog.y.astype(int),
                self.catalog.x.astype(int)]
        # Add a SinglePSF object for each source without any neighbors
        # and a group object for each source with neighbors that might
        # affect its flux
        for group_id in self.group_ids:
            group_idx = (group_id==group_indices)
            group_count = positions[group_idx].shape[0]
            if group_count==1:
                group_psf = astropyp.phot.psf.SinglePSF(
                    self.psf._psf_array,
                    amplitudes[group_idx][0],
                    positions[group_idx][0][0],
                    positions[group_idx][0][1],
                    self.psf._subsampling,
                    self.psf.fix_com
                )
            else:
                # Create PSF object for the entire group
                group_psf = astropyp.phot.psf.GroupPSF(
                    group_id, self.psf, positions[group_idx], 
                    amplitudes[group_idx], mask_img=self.mask_group,
                    show_plots=self.show_plots)
            self.groups[group_id] = group_psf
        if self.show_plots or verbose:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except ImportError:
                raise Exception(
                    "You must have matplotlib installed to create plots")
            fig, ax = plt.subplots()
            x = positions[:,0]
            y = positions[:,1]
            for group in self.group_ids:
                ax.plot(
                    x[group_indices==group], 
                    y[group_indices==group], 'o')
            plt.show()
        return self.groups
    
    def perform_psf_photometry(self, separation=None, 
            group_sources=True, verbose=False, fit_position=True,
            pos_range=0, indices=None):
        """
        Perform PSF photometry on all of the sources in the catalog,
        or if indices is specified, a subset of sources.
        
        Parameters
        ----------
        separation: float
            Separation (in pixels) for members to be considered
            part of the same group *Default=psf width*
        group_sources: bool
            Whether or not to group the sources (if groups have not
            already bee created for this SingleImage then this must
            be set to True). *Default=True*
        verbose: bool
            Whether or not to show info about the fit progress.
            *Default=False*
        fit_position: bool
            Whether or not to fit the position along with the
            amplitude of each source. *Default=True*
        pos_range: int
            Maximum number of pixels (in image pixels) that
            a sources position can be changed. If ``pos_range=0``
            no bounds will be set. *Defaul=0*
        indices: `~numpy.ndarray`
             Indices for sources to calculate PSF photometry.
             It is often advantageous to remove sources with
             bad pixels and sublinear flux to save processing time.
             All sources not included in indices will have their
             psf flux set to NaN.
        """
        # By default the sources are grouped before performing PSF
        # photometry
        if group_sources:
            self.create_psf_groups(separation=separation, verbose=verbose)
        
        if indices is not None:
            if isinstance(indices, six.string_types):
                indices = self.indices[indices]
            groups = np.unique(self.indices['group'][indices])
        else:
            groups = self.groups
        
        # Fit PSF for each group or isolated source
        psf_flux = np.zeros(self.catalog.shape[0])
        psf_flux[:] = np.nan
        positions = np.array(zip(self.catalog.x, self.catalog.y))
        group_indices = self.indices['group']
        data = self.img
        
        for group_id in groups:
            group = self.groups[group_id]
            if verbose:
                level = logger.getEffectiveLevel()
                logger.setLevel(logging.INFO)
                logger.info("Fitting {0}".format(group_id))
                logger.setLevel(level)
            if isinstance(group, astropyp.phot.psf.SinglePSF):
                amplitude = group.fit(data, fit_position, pos_range)
                flux = self.psf.get_flux(amplitude)
            elif isinstance(group, astropyp.phot.psf.GroupPSF):
                group_idx = (group_indices==group.group_id)
                amplitudes = np.array(group.fit(data))
                flux = self.psf.get_flux(amplitude)
            else:
                raise Exception("PSF photometry is currently only"
                    "supported for the SinglePSF and GroupPSF classes")
            psf_flux[group_indices==group_id] = np.array(flux)
        # Caluclate the error in the PSF flux and magnitude
        # In the future a better method may be to look at the
        # residual left over after the PSF is subtracted from
        # the background
        if self.gain is not None:
            psf_flux_err = 1.0857*np.sqrt(
                2*np.pi*self.psf._radius**2*self.bkg.globalrms**2+
                psf_flux/self.gain)
        else:
            psf_flux_err = 1.0857*np.sqrt(
                2*np.pi*self.psf._radius**2*self.bkg.globalrms**2
                )
        # Save the psf derived quantities in the catalog
        # Ignore divide by zero errors that occur when sources
        # have zero psf flux (i.e. bad sources)
        np_err = np.geterr()
        np.seterr(divide='ignore')
        psf_mag = -2.5*np.log10(psf_flux/self.exptime)
        self.catalog.sources['psf_flux'] = psf_flux
        self.catalog.sources['psf_flux_err'] = psf_flux_err
        self.catalog.sources['psf_mag'] = psf_mag
        self.catalog.sources['psf_mag_err'] = psf_flux_err/psf_flux
        np.seterr(**np_err)
        return psf_flux

def old_clean_sources(obs, mag_name, ref_name, check_columns=[], clipping=1):
    """
    Remove NaN values and clip sources outside a given number of standard 
    deviations
    
    Parameters
    ----------
    obs: structured array-like
        astropy.table.QTable, pandas.DataFrame, or structured array of 
        observations
    mag_name: str
        Name of the magnitude field
    ref_name: str
        Name of the reference catalog magnitude field
    check_columns: list of strings (optional)
        Names of columns to check for NaN values
    clipping: float (optional)
        Maximum number of standard deviations from the mean that a good 
        source will be found.
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
    Remove NaN values from a list of check_columns and clip based on the values 
    of certain error columns
    
    Parameters
    ----------
    obs: structured array-like
        astropy.table.Table, pandas.DataFrame, or structured array of 
        observations
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
            raise Exception(
                "You did not specify a value or list of values"
                " to use for clipping")
        if isinstance(clipping, float) or isinstance(clipping, int):
            clipping = [clipping for n in range(len(err_columns))]
        for n,col in enumerate(err_columns):
            good_sources = good_sources[good_sources[col]<clipping[n]]
    temp_check(good_sources, "error_columns")
    # Some sources have bad PSF fits so we check the the PSF 
    # magnitudes and Aperture Magnitudes are nearly the same
    good_sources = good_sources[
        np.abs(good_sources['MAG_AUTO']-good_sources['MAG_PSF'])<.02]
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

def match_all_catalogs(catalogs, ra_names, dec_names, 
        max_separation=1*apu.arcsec, min_detect=None, combine=True):
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

def calibrate_standard(sources, mag_name, ref1_name, ref2_name, mag_err_name, 
        ref1_err_name, ref2_err_name, init_zero=-25, init_color=-.1, 
        init_extinction=.1, fit_package='scipy', airmass_name='airmass'):
    """
    Calibrate a standard field with a set of refernce fields
    
    Parameters
    ----------
    sources: `astropy.table.QTable`
        Catalog of observations
    mag_name: str
        Name of the magniude column in ``sources``
    ref1_name: str
        Name of the reference column in ``sources`` in the same filter as 
        ``mag_name``
    ref2_name: str
        Name of the reference column in ``sources`` to use for the color 
        correction coefficient
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
        good_sources['color'] = good_sources[ref1_name] - \
            good_sources[ref2_name]
        result = smf.OLS.from_formula(formula='diff ~ color + airmass', 
            data=good_sources).fit()
        results = [result.params.Intercept, result.params.color, 
            result.params.airmass],result
    else:
        raise Exception(
            "fit_package must be either 'statsmodels' or 'scipy'(default)")
    logger.debug(
        "Zero point: {0}\nColor Correction: {1}\nExtinction: {2}\n".format
            (*results[0]))
    return results

def calibrate_2band(instr1, instr2, airmass1, airmass2, coeff1, coeff2,
        zero_key='zero', color_key='color', extinct_key='extinction'):
    """
    This solves the set of equations:
        i_0 = i + A_i + C_i(i-z) + k_i X
        z_0 = z + A_z + C_z(z-i) + k_z X
    where i_0 and z_0 are the instrumental magnitudes, A_i and A_z are the 
    zero points, C_i and C_z are the color terms, k_i and k_z are the 
    atmospheric coefficients, and X is the airmass.
    
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
        List of coeffients for calibrating instrumental magnitudes for 
        instrument 1.
            * coeff1[0]: zeropoint
            * coeff1[1]: color coeffcient
            * coeff1[2]: extinction coefficient
    coeff2: array-like
        List of coeffients for calibrating instrumental magnitudes for 
        instrument 2
    
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
    where Y0 is the instrumental magnitude, A_Y is the zero point, C_Y is the 
    color coefficent, k_Y is the extinction coefficient, and X is the airmass
    """
    if color_band is not None:
        mag = (instr - coeff[zero_key] + coeff[color_key]*color_band - 
            coeff[extinct_key]*airmass)/(1+coeff[color_key])
    else:
        mag = instr - coeff[zero_key] - coeff[color_key]*airmass
    return mag