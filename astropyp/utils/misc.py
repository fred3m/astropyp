import os
import logging

logger = logging.getLogger("astropyp.utils.misc")

class AstropypError(Exception):
    """
    Generic astropyp errors
    """
    pass

def str_2_bool(bool_str):
    """
    Case independent function to convert a string representation of a 
    boolean (``'true'``/``'false'``, ``'yes'``/``'no'``) into a ``bool``. This is case 
    insensitive, and will also accept part of a boolean string 
    (``'t'``/``'f'``, ``'y'``/``'n'``).
    
    Raises a :py:class:`astromatic.utils.pipeline.AstropypError` 
    if an invalid expression is entered.
    
    Parameters
    ----------
    bool_str: str
        String to convert to a boolean
    """
    lower_str = bool_str.lower()
    if 'true'.startswith(lower_str) or 'yes'.startswith(lower_str):
        return True
    elif 'false'.startswith(lower_str) or 'no'.startswith(lower_str):
        return False
    else:
        raise AstropypError(
            "'{0}' did not match a boolean expression "
            " (true/false, yes/no, t/f, y/n)".format(bool_str))

def get_bool(prompt):
    """
    Prompt a user for a boolean expression and repeat until a valid boolean
    has been entered.
    
    Parameters
    ----------
    prompt: str
        The text to prompt the user with.
    """
    try:
        bool_str = str_2_bool(raw_input(prompt))
    except AstropypError:
        print(
            "'{0}' did not match a boolean expression "
            "(true/false, yes/no, t/f, y/n)".format(bool_str))
        return get_bool(prompt)
    return bool_str

def check_path(pathname, path):
    """
    Check if a path exists and if it doesn't, give the user the option to create it.
    """
    if not os.path.exists(path):
        if get_bool("{0} '{1}' does not exist, create (y/n)?".format(pathname, path)):
            create_paths(path)
        else:
            raise DecamError("{0} does not exist".format(pathname))

def funpack_file(filename, fits_path):
    """
    Check if a file contains a compressed image and if it does decompress it and save it
    """
    logger.info("decompressing '{0}' to '{1}'\n".format(filename, fits_path))
    compressed = False
    hdulist = fits.open(filename)
    new_hdulist = fits.HDUList([hdulist[0]])
    for hdu in hdulist[1:]:
        if isinstance(hdu, fits.hdu.compressed.CompImageHDU):
            new_hdulist.append(fits.ImageHDU(hdu.data, hdu.header))
            compressed = True
        else:
            new_hdulist.append(hdu)
    if compressed:
        new_hdulist.writeto(fits_path, clobber=True)