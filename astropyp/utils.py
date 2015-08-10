# Copyright 2015 Fred Moolekamp
# BSD 3-clause license
import os
import logging
import pandas
import numpy as np
from astropy.table import Table, vstack, join
from astropy.coordinates import SkyCoord
import astropy.units as apu
from astropy.io import fits
from collections import OrderedDict
import logging

logger = logging.getLogger('decamtoyz.utils')

class DecamError(Exception):
    """
    Errors occuring in DECam tools
    """
    pass
    
def get_fits_name(expnum, prodtype):
    """
    Get the name of a fits image based on its exposure number and prodtype
    """
    if prodtype=='image':
        fits_name = '{0}.fits'.format(expnum)
    else:
        fits_name = "{0}.{1}.fits".format(expnum, prodtype)
    return fits_name

def funpack_file(filename, fits_path):
    """
    Funpack a compressed fits file and copy it to a new path
    """
    import subprocess
    import shutil
    if not os.path.isfile(fits_path):
        logger.info("funpacking '{0}' to '{1}'\n".format(filename, fits_path))
        compressed_name = fits_path.replace('.fits', '.fits.fz')
        shutil.copyfile(filename, compressed_name)
        subprocess.call('funpack '+compressed_name, shell=True)
        os.remove(compressed_name)
        logger.debug('remove {0}'.format(compressed_name))
        #subprocess.call('funpack '+filename, shell=True)
        #funpacked_name = os.path.basename(filename[:-3])
        #cmd = 'mv '+os.path.join(os.path.dirname(filename), funpacked_name)
        #cmd += ' '+fits_path
        #subprocess.call(cmd, shell=True)

def create_ahead(expnum, temp_path):
    """
    SExtractor doesn't copy info from the Primary HDU header into the catalog headers,
    so we need to copy important information about the images into the headers using
    an ahead file
    """
    filename = os.path.join(temp_path, expnum+'.fits')
    ahead_name = filename.replace('.fits', '.ahead')
    if not os.path.isfile(ahead_name):
        hdulist = fits.open(filename, memmap=True)
        f = open(ahead_name, 'w')
        obj = hdulist[0].header['OBJECT']
        instrument = hdulist[0].header['INSTRUME']
        mjd = hdulist[0].header['MJD-OBS']
        filtr = hdulist[0].header['FILTER']
        for n in range(61):
            f.write("OBJECT  = '"+str(obj)+'\n')
            f.write("INSTRUME= '"+str(instrument)+"'\n")
            f.write("MJD-OBS = "+str(mjd)+'\n')
            f.write("FILTER  = '"+str(filtr)+"'\n")
            f.write("END     \n")
        f.close()

def check_path(pathname, path):
    """
    Check if a path exists and if it doesn't, give the user the option to create it.
    """
    if not os.path.exists(path):
        import toyz.utils.core as core
        if core.get_bool("{0} '{1}' does not exist, create (y/n)?".format(pathname, path)):
            core.create_paths(path)
        else:
            raise DecamError("{0} does not exist".format(pathname))

def get_header_info(pipeline, cat_name, frame):
    """
    Get the exposure time, airmass, and gain from a DECam fits header
    
    Parameters
    ----------
    pipeline: :class:`.Pipeline`
        Object that contains parameters about a decam pipeline
    cat_name: str
        Name of a catalog that contains a list of sources found using decamtoyz
    frame: int
        CCD number of the header to load
    
    Returns
    -------
        exptime: float
            Exposure time of the image
        airmass: float
            Airmass (sec(zenith distance))
        gain: float
            Gain in electrons/adu
    """
    from decamtoyz.index import query_idx
    expnum = os.path.basename(cat_name).split('-')[0]
    sql = "select * from decam_files where EXPNUM={0} and PRODTYPE='image'".format(int(expnum))
    files = query_idx(sql, pipeline.idx_connect_str)
    header = fits.getheader(os.path.join(pipeline.paths['decam'], files.iloc[0].filename), ext=0)
    exptime = header['exptime']
    airmass = 1/np.cos(np.radians(header['ZD']))
    header = fits.getheader(os.path.join(pipeline.paths['decam'], files.iloc[0].filename), ext=frame)
    gain = header['arawgain']
    return exptime, airmass, gain

def get_idx_info(connection, conditions=None):
    """
    Get general information about a decam index
    
    Parameters
    ----------
    connection: str or `sqlalchemy.engine.base.Engine`
        If connection is a string then a sqlalchemy Engine will be created, otherwise
        ``connection`` is an sqlalchemy database engine that will be used for the query
    conditions: str (optional)
        Optional conditions to limit the scope of the index query (for example only
        get information for a given night or proposal)
    
    Returns
    -------
    idx_info: dict
        Dictionary containing information about the index
    """
    from decamtoyz.index import query_idx
    sql = "select * from decam_obs"
    if conditions is not None:
        sql += " where "+conditions
    exposures = query_idx(sql, connection)
    objects = exposures['object'].str.split('-').apply(pandas.Series).sort(0)[0].unique()
    proposals = exposures['propid'].unique()
    nights = exposures['cal_date'].unique()
    filters = pandas.unique(exposures['filter'].str[0])
    exptimes = {}
    for f in filters:
        exptimes[f] = exposures[exposures['filter'].str.startswith(f)]['exptime'].unique().tolist()
    
    idx_info = {
        'proposals': proposals,
        'objects': objects,
        'nights': nights,
        'filters': filters,
        'exptimes': exptimes
    }
    
    return idx_info

def get_exp_files(pipeline, night, obj, filtr, exptime, proctype='InstCal', 
        prep_files=False):
    """
    Get all of the exposures for a given night, object, and filter. This funpacks the
    relevant exposure fits image, weight maps, and data quality masks and copies them into the
    pipelines temp_path directory. 
    """
    from decamtoyz.index import query_idx
    sql = "select * from decam_obs where cal_date='{0}' and object like'{1}%'".format(night, obj)
    sql += "and object not like '%%SLEW' and filter like '{0}%' and exptime='{1}'".format(
        filtr, exptime
    )
    exposures = query_idx(sql, pipeline.idx_connect_str).sort(['expnum'])
    expnums = exposures['expnum'].unique()
    # Get the filenames for the image. dqmask, and wtmap products for each exposure
    exp_files = {expnum:{} for expnum in expnums}
    all_exposures = []
    for idx, row in exposures.iterrows():
        sql = "select * from decam_files where EXPNUM={0} and proctype='{1}'".format(
            row['expnum'], proctype)
        files = query_idx(sql, pipeline.idx_connect_str)
        for fidx, file in files.iterrows():
            fits_name = get_fits_name(file['EXPNUM'], file['PRODTYPE'])
            fits_path = os.path.join(pipeline.temp_path, fits_name)
            # Funpack the file
            if prep_files:
                funpack_file(file['filename'], fits_path)
            exp_files[file['EXPNUM']][file['PRODTYPE']] = fits_path
            if file['PRODTYPE']=='image':
                all_exposures.append(fits_path)
                # Create ahead files
                if prep_files:
                    create_ahead(file['EXPNUM'], pipeline.temp_path)
    return (exposures, expnums, exp_files, all_exposures)

def funpack_exposures(idx_connect_str, exposures, temp_path, proctype='InstCal'):
    """
    Run funpack to unpack compressed fits images (fits.fz) so that the
    AstrOmatic suite can read them. This also creates an ahead file for each
    image that can be used by SCAMP containing the appropriate header
    information
    
    Parameters
    ----------
    idx_connect_str: str
        Connection to database containing decam observation index
    exposures: `pandas.DataFrame`
        DataFrame with information about the observations
    temp_path: str
        Temporary directory where files are stored for pipeline processing
    proctype: str
        PROCTYPE from DECam image header (processing type of product from 
        DECam community pipeline)
    """
    from decamtoyz.index import query_idx
    expnums = exposures['expnum'].unique()
    # Get the filenames for the image, dqmask, and wtmap products for each exposure
    for idx, row in exposures.iterrows():
        sql = "select * from decam_files where EXPNUM={0} and proctype='{1}'".format(
            row['expnum'], proctype)
        files = query_idx(sql, idx_connect_str)
        for fidx, file_info in files.iterrows():
            fits_name = get_fits_name(file_info['EXPNUM'], file_info['PRODTYPE'])
            fits_path = os.path.join(temp_path, fits_name)
            funpack_file(file_info['filename'], fits_path)
            if file_info['PRODTYPE']=='image':
                create_ahead(file_info['EXPNUM'], temp_path)

def match_standard(pipeline, cat_names, ref_cat, filtr, ref_ra_name, ref_dec_name,
        ref_fields, frames=range(1,62), ignore_empty=False):
    """
    Match a list of catalogs to a reference catalog.
    
    Parameters
    ----------
    pipeline: :class:`~astromatic_wrapper.uitls.pipeline.Pipeline`
        Object that contains parameters about a decam pipeline
    cat_names: list of strings
        Names of catalog files to use in reduction
    ref_cat: str
        Path and filename of the reference catalog to use
    filtr: str
        Name of the filter to use
    ref_ra_name: str
        Name of the RA column in the reference catalog
    ref_dec_name: str
        Name of the DEC column in the reference catalog
    ref_fields: list or dict
        List or Dictionary of columns from the refernce catalog to include in the 
        sources catalog. If ref_fields is a list then the elements are the column names in
        the reference catalog and the column names in the sources catalog. If ref_fields is a
        dict then the keys are the names in the source catalog and the values are the names in
        the refernce catalog.
    frames: list of strings
        List of frame numbers (as strings) to match to the reference catalog
    ignore_empty: bool (optional)
        By default if there are no matches between the reference catalog and one of the
        source frames, an exception is raised. To ignore this exception set ``ignore_empty=True``.
    
    Returns
    -------
    sources: `pandas.DataFrame`
        Catalog of matched sources with both observed and reference magnitudes and errors
    """
    if not isinstance(ref_cat, Table):
        ref_cat = Table.read(ref_cat, hdu=2)
    if isinstance(ref_fields, list):
        ref_fields = OrderedDict([[col,col] for col in ref_fields])
    ref_coords = SkyCoord(ref_cat[ref_ra_name], ref_cat[ref_dec_name], unit='deg')
    # Calibrate each frame individually
    sources = None
    for frame in frames:
        logger.info('Loading sources for frame {0}'.format(frame))
        #sources = None
        for cat_name in cat_names:
            exptime, airmass, gain = get_header_info(pipeline, cat_name, frame)
            logger.info('{0}: airmass={1}, exptime={2}, filename={3}'.format(cat_name, airmass, 
                exptime, cat_name))
            cat_frame = Table.read(os.path.join(pipeline.paths['catalog'], cat_name), hdu=frame)
            cat_frame['FLUX_PSF'][cat_frame['FLUX_PSF'] * gain/exptime ==0] = np.nan
            cat_frame[filtr] = -2.5*np.log10(cat_frame['FLUX_PSF'] * gain/exptime)
            cat_frame['airmass'] = airmass
            cat_frame['filename'] = cat_name
            cat_frame['frame'] = frame
            #cat_frame['mag_err'] = 2.5*np.log10(1+cat_frame['FLUXERR_PSF']/cat_frame['FLUX_PSF'])
            
            # Find matches from reference catalog and add magnitudes to the catalog
            c1 = SkyCoord(cat_frame['XWIN_WORLD'], cat_frame['YWIN_WORLD'], unit='deg')
            idx, d2, d3 = c1.match_to_catalog_sky(ref_coords)
            ref_colnames = []
            ref_columns = []
            for colname, col in ref_fields.items():
                ref_colnames.append(colname)
                ref_columns.append(ref_cat[idx][col])
            cat_columns = [cat_frame['XWIN_WORLD'], cat_frame['YWIN_WORLD'],
                cat_frame['airmass'], cat_frame['filename'], cat_frame['frame'],
                cat_frame[filtr], cat_frame['MAG_PSF'], cat_frame['MAG_AUTO'],
                cat_frame['MAGERR_PSF'], d2.to('arcsec'), cat_frame['FLAGS'],
                cat_frame['FLAGS_WEIGHT']] + ref_columns
            cat_colnames = ['ra', 'dec', 'airmass', 'filename', 'ccd',
                filtr, 'psf_'+filtr, 'aper_'+filtr, 
                'err_'+filtr, 'separation', 'FLAGS', 'FLAGS_WEIGHT'] + ref_colnames
            catalog = Table(cat_columns, names=tuple(cat_colnames))
            
            # Only use entries where the sources are closer than 1 arcsec and add them to the total catalog
            good_catalog = catalog[d2.to('arcsec')<1*apu.arcsec]
            if len(good_catalog)==0:
                if ignore_empty:
                    import warnings
                    warnings.warn("No matching entries for {0}".format(cat_name))
                else:
                    raise Exception("No matching entries for {0}".format(cat_name))
            if sources is None:
                sources = good_catalog
            else:
                sources = vstack([sources, good_catalog])
    return sources