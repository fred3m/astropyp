class AstropypIoError(Exception):
    pass

def create_paths(paths):
    """                                                                         
    Search for paths on the server. If a path does not exist, create the necessary directories.
    For example, if ``paths=['~/Documents/images/2014-6-5_data/']`` and only the path 
    *'~/Documents'* exists, both *'~/Documents/images/'* and 
    *'~/Documents/images/2014-6-5_data/'* are created.
    
    Parameters
    ----------
    paths: str or list of strings
        If paths is a string, this is the path to search for and create. 
        If paths is a list, each one is a path to search for and create
    """
    from astropy.extern.six import string_types
    if isinstance(paths, string_types):
        paths=[paths]
    for path in paths:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise AstropypIoError("Problem creating new directory, check user permissions")

def check_path(path, auto_create=False):
    """
    Check if a path exists and if it doesn't, give the user the option to create it.
    
    Parameters
    ----------
    path: str
        Name of the path to check
    auto_create: bool (optional)
        If the path does not exist and ``auto_create==True``, the path will 
        automatically be created, otherwise the user will be prompted to create the path
    """
    from astropyp.utils.misc import get_bool
    if not os.path.exists(path):
        if auto_create or get_bool("'{0}' does not exist, create (y/n)?".format(path)):
            create_paths(path)
        else:
            raise AstropypIoError("{0} does not exist".format(path))