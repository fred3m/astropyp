# Copyright 2015 Fred Moolekamp
# BSD 3-clause license
"""
Functions to convert FITS files or astropy Tables to FITS_LDAC files and
vice versa.
"""

def convert_hdu_to_ldac(hdu):
    """
    Convert an hdu table to a fits_ldac table (format used by astromatic suite)
    
    Parameters
    ----------
    hdu: `astropy.io.fits.BinTableHDU` or `astropy.io.fits.TableHDU`
        HDUList to convert to fits_ldac HDUList
    
    Returns
    -------
    tbl1: `astropy.io.fits.BinTableHDU`
        Header info for fits table (LDAC_IMHEAD)
    tbl2: `astropy.io.fits.BinTableHDU`
        Data table (LDAC_OBJECTS)
    """
    from astropy.io import fits
    import numpy as np
    tblhdr = np.array([hdu.header.tostring(',')])
    col1 = fits.Column(name='Field Header Card', array=tblhdr, format='13200A')
    cols = fits.ColDefs([col1])
    tbl1 = fits.BinTableHDU.from_columns(cols)
    tbl1.header['TDIM1'] = '(80, {0})'.format(len(hdu.header))
    tbl1.header['EXTNAME'] = 'LDAC_IMHEAD'
    tbl2 = fits.BinTableHDU(hdu.data)
    tbl2.header['EXTNAME'] = 'LDAC_OBJECTS'
    return (tbl1, tbl2)

def convert_table_to_ldac(tbl):
    """
    Convert an astropy table to a fits_ldac
    
    Parameters
    ----------
    tbl: `astropy.table.Table`
        Table to convert to ldac format
    Returns
    -------
    hdulist: `astropy.io.fits.HDUList`
        FITS_LDAC hdulist that can be read by astromatic software
    """
    from astropy.io import fits
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix='.fits', mode='rb+')
    tbl.write(f, format='fits')
    f.seek(0)
    hdulist = fits.open(f, mode='update')
    tbl1, tbl2 = convert_hdu_to_ldac(hdulist[1])
    new_hdulist = [hdulist[0], tbl1, tbl2]
    new_hdulist = fits.HDUList(new_hdulist)
    return new_hdulist

def save_table_as_ldac(tbl, filename, **kwargs):
    """
    Save a table as a fits LDAC file
    
    Parameters
    ----------
    tbl: `astropy.table.Table`
        Table to save
    filename: str
        Filename to save table
    kwargs:
        Keyword arguments to pass to hdulist.writeto
    """
    hdulist = convert_table_to_ldac(tbl)
    hdulist.writeto(filename, **kwargs)

def get_table_from_ldac(filename, frame=1):
    """
    Load an astropy table from a fits_ldac by frame (Since the ldac format has column 
    info for odd tables, giving it twce as many tables as a regular fits BinTableHDU,
    match the frame of a table to its corresponding frame in the ldac file).
    
    Parameters
    ----------
    filename: str
        Name of the file to open
    frame: int
        Number of the frame in a regular fits file
    """
    from astropy.table import QTable
    if frame>0:
        frame = frame*2
    tbl = QTable.read(filename, hdu=frame)
    return tbl

def isldac(filename):
    """
    Determine if a bintable is a FITS of FITS_LDAC file
    """
    from astropy.io import fits
    hdulist = fits.open(filename, memmap=True)
    if (hdulist[0].header['EXTEND'] and 'EXTNAME' in hdulist[1].header and 
            'LDAC' in hdulist[1].header['EXTNAME']):
        return True
    return False

def get_fits_table(filename, frame=1):
    """
    Get a table from a FITS file or FITS_LDAC file (autocheck the type)
    """
    if isldac(filename):
        tbl = get_table_from_ldac(filename, frame)
    else:
        from astropy.table import Table
        tbl = Table.read(filename, hdu=frame)
    return tbl

def get_fits_table_count(filename):
    """
    Get the number of tables in a FITS or FITS_LDAC files.
    """
    from astropy.io import fits
    hdulist = fits.open(filename, memmap=True)
    count = len(hdulist)-1
    if isldac(filename):
        return int(count/2)
    else:
        return count

def get_combined_table(filename, frames=None):
    """
    Combine all of the tables in a FITS or FITS_LDAC file into a single table
    """
    from astropy.table import vstack
    if frames is None:
        frames = range(1, get_fits_table_count(filename)+1)
    all_frames = None
    for frame in frames:
        tbl = get_fits_table(filename, frame)
        tbl['frame'] = frame
        if all_frames is None:
            all_frames = tbl
        else:
            all_frames = vstack([all_frames, tbl])
    return all_frames
            