import logging
import astropy.units as apu

logger = logging.getLogger('astropyp.instruments.decam.phot')

class DecamPhotError(Exception):
    pass

def get_phot_params(catalog, mag_name, img_filename=None, expnum=None, proctype='InstCal', 
        connection=None, frame_colname='frame', flux_name='FLUX_PSF'):
    """
    Get necessary parameters like airmass and exposure time from the FITS image headers
    """
    import numpy as np
    from astropyp import index
    from astropy.io import fits
    
    # Get fields from FITS header
    if img_filename is None:
        sql = "select * from decam_obs where EXPNUM={0} and PRODTYPE='image'".format(expnum)
        sql += " and PROCTYPE='{0}'".format(proctype)
        files = index.query(sql, connection)
        # Make sure there is only one image with the correct expnum
        if len(files)==0:
            raise DecamPhotError("No files found with EXPNUM {0}".format(expnum))
        elif len(files)>1:
            raise DecamPhotError("Multiple files found with EXPNUM {0}".format(expnum))
        img_filename = files[0]['filename']
    hdulist = fits.open(img_filename, memmap=True)
    header = hdulist[0].header
    exptime = header['EXPTIME']
    
    catalog['airmass'] = 1/np.cos(np.radians(header['ZD']))
    catalog[mag_name] = np.nan
    #catalog['filename'] = catalog_filename
    
    frames = np.unique(catalog[frame_colname])
    # add fields to catalog frame by frame
    for frame in frames:
        gain = hdulist[frame].header['arawgain']
        logger.debug('frame {0}: gain={1}, exptime={2}'.format(frame, gain, exptime))
        frame_condition = catalog[frame_colname]==frame
        counts = np.array(catalog[flux_name]) * gain/exptime
        catalog[flux_name][frame_condition & (counts==0)] = np.nan
        catalog[mag_name][frame_condition] = -2.5*np.log10(counts[frame_condition])
    catalog['ccd'] = catalog['frame']
    return catalog