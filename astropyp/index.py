"""
Build or load an index of decam files
"""
import os
import logging
import warnings

logger = logging.getLogger('astropyp.index')

def add_tbl(connection, columns=None, tbl_name='obs', echo=False):
    """
    Create or clear a set of tables for a decam file index and create the decam_keys
    table to link decam headers to table columns
    
    Parameters
    ----------
    connection: str or `~sqlalchemy.engine.base.Engine`
        Connection to the index database
    columns: list of tuples
        Columns to create in the table. This should always be a list of 
        tuples with 3 entries:
        
            1. column name (str)
            2. column data type (sqlalchemy data type, for example `Integer`)
            3. dictionary of optional parameters
        
        Example: ``[('EXPNUM', Integer, {'index': True}),('RA', Float, {})]``
    tbl_name: str
        Name of the table to create. If the table aleady exists nothing will be done
    echo: bool (optional)
        For debugging purposes you may wish to view the commands being sent
        from python to the database, in which case you should set ``echo=True``.
    
    Returns
    -------
    table_exists: bool
        If the table already exists ``True`` is returned, otherwise ``False`` is returned
    """
    from sqlalchemy import create_engine, MetaData, Table, Column, ForeignKey
    from sqlalchemy import Integer, Float, String
    
    engine = create_engine(connection, echo=echo)
    meta = MetaData()
    
    # check if the table name already exists
    if tbl_name in meta.tables.keys():
        warnings.warn('Table "{0}" already exists'.format(tbl_name))
        return True
    
    # Default column names for DECam (remove later)
    default_columns = [
        ('id', Integer, {'primary_key':True}),
        ('EXPNUM', Integer, {'index':True}),
        ('EXPTIME', Float, {}),
        ('DATE-OBS', Integer, {}),
        ('MJD-OBS', Float, {}),
        ('OBS-LAT', String, {}),
        ('OBS-LONG', String, {}),
        ('PROPID', String, {}),
        ('RA', String, {}),
        ('DEC', String, {}),
        ('FILTER', String, {}),
        ('OBSTYPE', String, {}),
        ('OBJECT', String, {}),
        ('DTCALDAT', String, {}),
        ('filename', String, {'index': True}),
        ('PROCTYPE', String, {'index': True}),
        ('PRODTYPE', String, {'index': True}),
    ]
    
    # Include default columns if the user keys included a '*' or if the user 
    # didn't specify any header keys
    if columns is None:
        columns = default_columns
    elif '*' in columns:
        columns = default_columns+columns
    
    # Create the table
    tbl_columns = [Column(col[0],col[1],**col[2]) for col in columns]
    tbl = Table(tbl_name, meta, *tbl_columns)
    tbl.create(engine, checkfirst=True)
    return False

def valid_ext(filename, extensions):
    """
    Check if a filename ends with a valid extension
    """
    from astropy.extern import six
    if isinstance(extensions, six.string_types):
        extensions = [extensions]
    if any([filename.endswith(ext) for ext in extensions]):
        return True
    return False

def get_files(path):
    """
    Get all the files in a directory
    """
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

def add_files(connection, tbl_name, filenames=None, paths=None, recursive=False, 
        no_warnings=True, column_func=None, extensions=['.fits','.fits.fz']):
    """
    Add fits header information to a database table.
    
    Parameters
    ----------
    connection: str or `~sqlalchemy.engine.base.Engine`
        Connection to the index database
    tbl_name: str
        Name of the table to add the files
    filenames: str or list (optional)
        Filename or list of filenames to add to the table
    paths: str or list (optional)
        Path or list of paths to search for files to add to the table. Only
        files that end with one of the ``extensions`` will be added.
    recursive: bool (optional)
        If ``recursive==True`` each path in ``paths`` will be recursively searched
        for files that end with one of the ``extensions``. The default value is
        ``False``.
    no_warnings: bool (optional)
        Often very old (or very new) fits files generate warnings in astropy.
        It is often useful to ignore these so the default is to ignore warnings
        (``no_warnings==True``).
    column_func: function (optional)
        It may be useful to calculate certain quantities like airmass that might not
        be in the fits header and store them in the table. If a ``column_func`` is
        passed to ``add_files`` it should receive the parameters ``hdulist`` and
        ``header_values``, a dictionary of all of the header values to be written
        to database. Any new keys should be added to ``header_values`` and the
        function should also return the modified ``header_values`` variable.
    extensions: list (optional)
        List of file extensions to search for in ``paths``. The default
        is ``extensions=['.fits','.fits.fz']
    
    Returns
    -------
    new_files: list
        List of files added to the database
    duplicates: list
        List of files already contained in the database
    """
    from astropy.io import fits
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy.orm import sessionmaker
    from astropy.extern import six
    
    # Sometimes older (or newer) fits files generate warnings by astropy, which a user
    # probably wants to suppress while building the image index
    if no_warnings:
        import warnings
        warnings.filterwarnings("ignore")
    
    # Open connection to database and load the table information
    engine = create_engine(connection)
    conn = engine.connect()
    meta = MetaData()
    meta.reflect(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get information about objects in the database (to prevent duplicate entries)
    tbl = meta.tables[tbl_name]
    old_filepaths = [f[0] for f in session.query(tbl.columns.filename).all()]
    session.close()
    
    # Search all filepaths for fits files
    if paths is not None:
        import itertools
        if isinstance(paths, six.string_types):
            paths = [paths]
        logger.info('paths: {0}'.format(paths))
        filepaths = []
        for path in paths:
            if recursive:
                filepaths.append(list(itertools.chain.from_iterable(
                    [[os.path.join(root, f) for f in files if valid_ext(f, extensions)] 
                        for root,dirs,files in os.walk(path)])))
            else:
                filepaths.append(get_files(path))
        # Flatten the list of lists
        filepaths = list(itertools.chain.from_iterable(filepaths))
    else:
        filepaths = []
    
    # Add any files specified by the user
    if filenames is not None:
        if isinstance(filenames, six.string_types):
            filenames = [filenames]
        filepaths = filepaths+filenames
    
    duplicates = []
    new_files = []
    # Iterate through paths to look for files to add
    for n, filepath in enumerate(filepaths):
        logger.info('{0} of {1}: {2}'.format(n, len(filepaths),filepath))
        if filepath not in old_filepaths:
            # Add header keys
            hdulist = fits.open(filepath, memmap=True)
            header_values = {}
            for col in tbl.columns:
                key = col.name
                if key in hdulist[0].header:
                    header_values[key] = hdulist[0].header[key]
                elif len(hdulist)>1 and key in hdulist[1].header:
                    header_values[key] = hdulist[1].header[key]
            header_values['filename'] = filepath
            
            # Run a code to calculate custom columns 
            #(for example if airmass is not in the header)
            if column_func is not None:
                header_values = column_func(hdulist, header_values)
            
            #Insert values into the table
            ins = tbl.insert().values(**header_values)
            conn.execute(ins)
            new_files.append(filepath)
        else:
            logger.info('duplicate: {0}'.format(filepath))
            duplicates.append(filepath)
    
    logger.debug('New files added: \n{0}'.format(new_files))
    logger.info('All duplicates: \n{0}'.format(duplicates))
    
    return new_files, duplicates

def query(sql, connection):
    """
    Query the index
    
    Parameters
    ----------
    sql: str
        SQL expression to execute on the database
    connection: str or `~sqlalchemy.engine.base.Engine`
        Connection to the index database
    
    Returns
    -------
    result: `~astropy.table.QTable`
    """
    from astropy.extern import six
    from astropy.table import QTable
    
    if isinstance(connection, six.string_types):
        from sqlalchemy import create_engine
        engine = create_engine(connection)
    else:
        engine = connection
    connection = engine.connect()
    result = connection.execute(sql)
    col_names = result.keys()
    result_list = result.fetchall()
    if len(result_list)>0:
        return QTable(rows=result_list, names=col_names)
    return QTable()