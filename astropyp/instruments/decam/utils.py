import os

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
    from astropyp import index
    from astropy.io import fits
    import numpy as np
    
    expnum = os.path.basename(cat_name).split('-')[0]
    sql = "select * from decam_files where EXPNUM={0} and PRODTYPE='image'".format(int(expnum))
    files = index.query(sql, pipeline.idx_connect_str)
    header = fits.getheader(os.path.join(pipeline.paths['decam'], files.iloc[0].filename), ext=0)
    exptime = header['exptime']
    airmass = 1/np.cos(np.radians(header['ZD']))
    header = fits.getheader(os.path.join(pipeline.paths['decam'], files.iloc[0].filename), ext=frame)
    gain = header['arawgain']
    return exptime, airmass, gain