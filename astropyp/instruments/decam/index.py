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
    from astropyp import index
    import pandas
    
    sql = "select * from decam_obs"
    if conditions is not None:
        sql += " where "+conditions
    exposures = index.query(sql, connection)
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
    from astropyp import index
    sql = "select * from decam_obs where cal_date='{0}' and object like'{1}%'".format(night, obj)
    sql += "and object not like '%%SLEW' and filter like '{0}%' and exptime='{1}'".format(
        filtr, exptime
    )
    exposures = index.query(sql, pipeline.idx_connect_str).sort(['expnum'])
    expnums = exposures['expnum'].unique()
    # Get the filenames for the image. dqmask, and wtmap products for each exposure
    exp_files = {expnum:{} for expnum in expnums}
    all_exposures = []
    for idx, row in exposures.iterrows():
        sql = "select * from decam_files where EXPNUM={0} and proctype='{1}'".format(
            row['expnum'], proctype)
        files = index.query(sql, pipeline.idx_connect_str)
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
    from astropyp import index
    expnums = exposures['expnum'].unique()
    # Get the filenames for the image, dqmask, and wtmap products for each exposure
    for idx, row in exposures.iterrows():
        sql = "select * from decam_files where EXPNUM={0} and proctype='{1}'".format(
            row['expnum'], proctype)
        files = index.query(sql, idx_connect_str)
        for fidx, file_info in files.iterrows():
            fits_name = get_fits_name(file_info['EXPNUM'], file_info['PRODTYPE'])
            fits_path = os.path.join(temp_path, fits_name)
            funpack_file(file_info['filename'], fits_path)
            if file_info['PRODTYPE']=='image':
                create_ahead(file_info['EXPNUM'], temp_path)