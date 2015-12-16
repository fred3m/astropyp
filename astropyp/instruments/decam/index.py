class AstropypDecamIdxErr(Exception):
    pass

def get_decam_default_columns():
    """
    Get a list of tuples that can be passed to `~astropyp.index.add_tbl`_ to
    create a new table for DECam files for the most common header fields to
    include in a search.
    """
    from sqlalchemy import Integer, Float, String
    
    # Default column names for DECam (remove later)
    default_columns = [
        ('id', Integer, {'primary_key':True}),
        ('EXPNUM', Integer, {'index':True}),
        ('EXPTIME', Float, {}),
        ('DATE-OBS', String, {}),
        ('MJD-OBS', Float, {}),
        ('OBS-LAT', String, {}),
        ('OBS-LONG', String, {}),
        ('OBSTYPE', String, {}),
        ('PROPID', String, {}),
        ('RA', String, {}),
        ('DEC', String, {}),
        ('FILTER', String, {'index': True}),
        ('OBSTYPE', String, {}),
        ('OBJECT', String, {'index': True}),
        ('DTCALDAT', String, {'index': True}),
        ('filename', String, {'index': True}),
        ('PROCTYPE', String, {}),
        ('PRODTYPE', String, {}),
    ]
    return default_columns

def add_stacks_to_idx(connection, obs_tbl, stack_tbl='temp_stacks',no_warnings=True):
    """
    Add Stacks created by the DECam community pipeline to the decam index database.
    
    .. note:: Entries will already exist for the stacks but they will not be
    associated with EXPNUMs, since each stack is made from multiple exposures.
    This step removes the old entries and replaces them with a new entry for
    each EXPNUM.
    
    To prevent files from being lost from the index during an error this
    process is run in a few steps. The first step is to create a new "temp_stacks" 
    table where an entries are made for each EXPNUM for each stack. 
    Then the original stack entries (with no EXPNUMs) are deleted from the table.
    Next the new entires from "temp_stacks" are copied back to the obs table.
    The "temp_stack" table is then deleted.
    
    Parameters
    ----------
    connection: str or `~sqlalchemy.engine.base.Engine`
        Connection to the index database
    obs_tbl: str
        Name of the table containing the decam observation info
    stack_tbl: str (optional)
        Name of the temporary table to store the expanded stacks. There is no
        need to change this unless a table with the name "temp_stacks" already
        exists in the database.
    """
    from astropy.table import QTable
    from astropy.io import fits
    import astropyp.index
    from sqlalchemy import MetaData
    from sqlalchemy.orm import sessionmaker
    
    # Sometimes older (or newer) fits files generate warnings by astropy, which a user
    # probably wants to suppress while building the image index
    if no_warnings:
        import warnings
        warnings.filterwarnings("ignore")
    
    # Create temporary table that is a duplicate
    engine = astropyp.index.init_connection(connection)
    meta = MetaData()
    meta.reflect(engine)
    tbl = meta.tables[obs_tbl]
    temp_tbl = astropyp.index.clone_tbl(tbl, meta, stack_tbl)
    meta.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Add a new entry for each expnum in the temporary table
    stack_entries = session.query(tbl).filter(tbl.columns.PROCTYPE=='Stacked').all()
    for row in stack_entries:
        hdulist = fits.open(row.filename)
        exps = [int(v[6:].lstrip('0')) for k,v in hdulist[0].header.items() 
            if k.startswith('IMCMB')]
        exp_values = dict(zip(row.keys(), row))
        del exp_values['id']
        for expnum in exps:
            exp_values['EXPNUM'] = expnum
            session.execute(temp_tbl.insert().values(**exp_values))
            session.commit()
    
    # Remove all entries where the expnum is None
    session.execute(tbl.delete().where(tbl.columns.EXPNUM==None))
    session.commit()
    
    # Copy all entries from the stack table to the observation table
    stack_entries = session.query(temp_tbl).all()
    
    for row in stack_entries:
        exp_values = dict(zip(row.keys(), row))
        del exp_values['id']
        session.execute(tbl.insert().values(**exp_values))
        session.commit()
    
    # Drop the temporary table
    session.close()
    temp_tbl.drop(engine)

def prep_exp_files(night=None, obj=None, filtr=None, exptime=None, proctype=None,
        pipeline=None, connection=None, obs_tbl_name=None):
    """
    Get all of the exposures for a given night, object, and filter. This funpacks the
    relevant exposure fits image, weight maps, and data quality masks and copies them into the
    pipelines temp_path directory. 
    """
    import astropyp.index
    from astropy.table import QTable
    import numpy as np
    from sqlalchemy import or_, and_, create_engine, MetaData
    from sqlalchemy.orm import sessionmaker
    
    if (connection is None or obs_tbl_name is None) and pipeline is None:
        raise AstropypDecamIdxErr(
            'You must either specify a decam pipeline or a connection and tbl')
    if connection is None:
        connection = pipeline.connection
    if obs_tbl_name is None:
        obs_tbl_name = pipeline.obs_tbl_name
    
    meta, obs_tbl, session = astropyp.index.connect2idx(connection, obs_tbl_name)
    
    # Get relevant exposures
    filters = {
        'DTCALDAT': obs_tbl.columns.DTCALDAT==night,
        'EXPTIME': obs_tbl.columns.EXPTIME==exptime,
        'PROCTYPE': obs_tbl.columns.PROCTYPE==proctype,
        'OBJECT': obs_tbl.columns.OBJECT.like("{0}%".format(obj))
    }
    filters = []
    if obj is not None:
        filters.append(obs_tbl.columns.OBJECT.like("{0}%".format(obj)))
    if night is not None:
        filters.append(obs_tbl.columns.DTCALDAT==night)
    if filtr is not None:
        filters.append(obs_tbl.columns.FILTER.like("{0}%".format(filtr)))
    if exptime is not None:
        filters.append(obs_tbl.columns.EXPTIME==exptime)
    if proctype is not None:
        filters.append(obs_tbl.columns.PROCTYPE==proctype)
    
    #exp_result = session.query(obs_tbl).filter(and_(
    #    obs_tbl.columns.DTCALDAT==night, 
    #    obs_tbl.columns.FILTER.like("{0}%".format(filtr)),
    #    obs_tbl.columns.EXPTIME==exptime,
    #    obs_tbl.columns.PROCTYPE==proctype,
    #    obs_tbl.columns.OBJECT.like("{0}%".format(obj))
    #)).all()
    exp_result = session.query(obs_tbl).filter(and_(*filters))
    exposures = QTable(rows=[x for x in exp_result], names=meta.tables[obs_tbl_name].columns.keys())
    return exposures

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