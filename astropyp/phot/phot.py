import logging
import warnings
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
    catalog: `~astropyp.catalog.Catalog` or `~astropy.table.Table`
        Catalog of sources with mappings to standard catalog fields.
        See `~astropyp.catalog.Catalog` for more on Catalogs.
        If an `~astropy.table.Table` is passed instead it will be
        converted to an Catalog with the standard mapping
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
        
        if not isinstance(self.catalog, astropyp.catalog.Catalog):
            self.catalog = astropyp.catalog.Catalog(self.catalog)
        # If the image is part of an focal array from a larger exposure,
        # use the exposure settings to set parameters
        if self.exptime is None and self.exposure is not None:
            self.exptime = exposure.exptime
        
        self.group_ids = range(len(self.groups))
    
    def detect_sources(self, sex_params={}, aper_radius=None, 
            subtract_bkg=False, gain=None, wcs=None, exptime=None,
            windowed=True, edge_val=1, transform='wcs'):
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
            wcs, aper_radius, windowed, edge_val, transform)
        sources, self.bkg = result
        self.catalog = astropyp.catalog.Catalog(
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
        Group sources with overlapping PSF's. This method is no longer
        used and will likely be depreciated as long as the other
        crowded field routines work.
        """
        warnings.warn("This function is depreciated and will be removed "
            "in the future")
        
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

class Exposure:
    """
    Container for a focal array of SingleImages, for example a camera image
    with multiple CCD images.
    
    Parameters
    ----------
    exp_info: dict
        Dictionary of parameters for the exposure. At a minimum this should
        include the exposure time and airmass of the exposure
    """
    def __init__(self, exp_info, 
            img_filename=None, dqmask_filename=None, wtmap_filename=None,
            img=None, dqmask=None, wtmap=None, wcs=None, 
            detect_sources=True, frames=None, memmap=True,
            aper_radius=None, subsampling=5, psf=None, sex_params={}, 
            gain=None):
        # Set exposure info as attributes of the class
        for k,v in exp_info.items():
            setattr(self, k, v)
        # Load the image data (if necessary)
        if img is None:
            img = fits.open(img_filename, memmap=memmap)
        if dqmask is None and dqmask_filename is not None:
            dqmask = fits.open(dqmask_filename, memmap=memmap)
        if wtmap is None and wtmap_filename is not None:
            wtmap = fits.open(wtmap_filename, memmap=memmap)
        self.img = img
        self.dqmask = dqmask
        self.wtmap = wtmap
        self.aper_radius = aper_radius
        self.sex_params = sex_params
        self.ccd_dict = OrderedDict()
        self.gain = gain
        
        # If the user didn't specify which frames to include, use all of them
        if frames is None:
            frames = range(1,len(img))
        
        # Add the CCD's to the exposure
        for frame in frames:
            img_data = img[frame].data.byteswap(True).newbyteorder()
            if dqmask is not None:
                dqmask_data = dqmask[frame].data.byteswap(True).newbyteorder()
            else:
                dqmask_data = None
            if wtmap is not None:
                wtmap_data = wtmap[frame].data.byteswap(True).newbyteorder()

            self.ccd_dict[frame] = DecamCCD(self, img[frame].header, img_data,
                dqmask_data, wtmap_data, aper_radius, subsampling, wcs)
        
        if detect_sources:
            for frame in frames:
                self.ccd_dict[frame].detect_sources(self.sex_params, 
                    gain=self.gain, aper_radius=self.aper_radius)

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