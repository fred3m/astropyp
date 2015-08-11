"""
Build or load an index of decam files
"""
from __future__ import division,print_function
import os
from collections import OrderedDict

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, backref
Base = declarative_base()
from sqlalchemy.orm import sessionmaker

# Map from header keys to table column names
hdr_keys = {
    'EXPNUM': 'expnum',
    'EXPTIME': 'exptime',
    'DATE-OBS': 'date_obs',
    'MJD-OBS': 'mjd_obs',
    'OBS-LAT': 'obs_lat',
    'OBS-LONG': 'obs_long',
    'PROPID': 'propid',
    'RA': 'ra',
    'DEC': 'dec',
    'FILTER': 'filter',
    'OBSTYPE': 'obstype',
    'OBJECT': 'object',
    'DTCALDAT': 'cal_date' # Calendar date
}

class DecamKeys(Base):
    """
    Typically dashes are not used in table names, and it would have been a bit of work to
    get them working in sqlalchemy. Instead, decam_keys is a map of decam header keys to 
    table name keys.
    """
    __tablename__ = 'decam_keys'
    id = Column(Integer, primary_key=True)
    hdr_key = Column(String, index=True, unique=True)
    table_key = Column(String, index=True)

class DecamObs(Base):
    """
    Table containing information about a decam observation (unique to an EXPNUM header
    keyword)
    """
    __tablename__ = 'decam_obs'
    id = Column(Integer, primary_key=True)
    expnum = Column(String, index=True)
    exptime = Column(String)
    date_obs = Column(String)
    mjd_obs = Column(String)
    obs_lat = Column(String)
    obs_long = Column(String)
    propid = Column(String)
    ra = Column(String)
    dec = Column(String)
    filter = Column(String)
    obstype = Column(String)
    object = Column(String)
    cal_date = Column(String)

class DecamFiles(Base):
    """
    Table containing data for all decam files
    """
    __tablename__ = 'decam_files'
    id = Column(Integer, primary_key=True)
    filename = Column(String, index=True)
    PROCTYPE = Column(String, index=True)
    PRODTYPE = Column(String, index=True)
    EXPNUM = Column(String, ForeignKey('decam_obs.expnum'), index=True)
    decam_obs = relationship(DecamObs, backref=backref('decam_files', uselist=True))

def create_idx(connection=None):
    """
    Create or clear a set of tables for a decam file index and create the decam_keys
    table to link decam headers to table columns
    """
    # Init sql alquemy session to add tables
    if connection is None:
        ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        connection = 'sqlite:///'+os.path.join(ROOT_DIR,'decam.db')
    engine = create_engine(connection)
    Base.metadata.reflect(engine)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    # If any of the tables already exist, delete all of the rows
    if engine.has_table('decam_keys'):
        session.query(DecamKeys).delete()
    if engine.has_table('decam_files'):
        session.query(DecamFiles).delete()
    if engine.has_table('decam_obs'):
        session.query(DecamObs).delete()
    # Typically dashes are not used in table names, and it would have been a bit of work to
    # get them working in sqlalchemy. Instead, decam_keys is a map of decam header keys to 
    # table name keys.
    for k,v in hdr_keys.items():
        decam_key = DecamKeys(hdr_key=k, table_key=v)
        session.add(decam_key)
    session.commit()
    session.close()

def get_dirs(path):
    return [p for p in os.listdir(path) if os.path.isdir(os.path.join(path,p))]

def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

def build(img_path='.', connection=None, create_new=False, recursive=False, no_warnings=True):
    """
    Given a path to a set of decamfiles
    """
    import astropy.io.fits as pyfits
    # Sometimes older (or newer) fits files generate warnings by astropy, which a user
    # probably wants to suppress while building the image index
    if no_warnings:
        import warnings
        warnings.filterwarnings("ignore")
        
    if create_new:
        create_idx(connection)
    # Walk through the given path and add all of the files to the index
    if recursive:
        paths = os.walk(img_path)
    else:
        paths = [[img_path, get_dirs(img_path), get_files(img_path)]]
    # Open connection to database
    if connection is None:
        ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        connection = 'sqlite:///'+os.path.join(ROOT_DIR,'decam.db')
    engine = create_engine(connection)
    Base.metadata.reflect(engine)
    Base.metadata.create_all(engine)
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    exposures = []
    # Iterate through paths to look for files to add
    for root, dirs, files in paths:
        print('root:', root)
        for n, filename in enumerate(files):
            print(n, filename)
            if filename.endswith('.fits.fz'):
                #print(filename)
                # Add the file to the index
                filepath = os.path.join(root, filename)
                hdulist = pyfits.open(filepath)
                # If the exposure has not been added to observations yet, add it now
                if hdulist[0].header['EXPNUM'] not in exposures :
                    exposures.append(hdulist[0].header['EXPNUM'])
                    hdr = {}
                    for hdr_key, tbl_key in hdr_keys.items():
                        if hdr_key in hdulist[0].header:
                            hdr[tbl_key] = hdulist[0].header[hdr_key]
                        elif hdr_key in hdulist[1].header:
                            hdr[tbl_key] = hdulist[1].header[hdr_key]
                    decam_obs = DecamObs(**hdr)
                    session.add(decam_obs)
                # Add the file info to the table
                kwargs = {
                    'PROCTYPE': hdulist[0].header['PROCTYPE'],
                    'PRODTYPE': hdulist[0].header['PRODTYPE']
                }
                kwargs['filename'] = filepath
                # If the image is a stack, add all of the exposures combined in the stack,
                # otherwise just add the EXPNUM field
                if hdulist[0].header['PROCTYPE'] == 'Stacked':
                    exps = [v[6:].lstrip('0') for k,v in hdulist[0].header.items() 
                        if k.startswith('IMCMB')]
                else:
                    exps = [hdulist[0].header['EXPNUM']]
                for exp in exps:
                    kwargs['EXPNUM'] = exp
                    decam_file = DecamFiles(**kwargs)
                    session.add(decam_file)
            else:
                print('Skipping', filename,'\n\n')
    session.commit()
    session.close()

def query(sql='select * from decam_files where PROCTYPE=="InstCal"', connection=None):
    """
    Query a decam index
    
    Parameters
    ----------
    sql: str
        Query to perform on the decam index database
    connection: str or `sqlalchemy.engine.base.Engine`
        If connection is a string then a sqlalchemy Engine will be created, otherwise
        ``connection`` is an sqlalchemy database engine that will be used for the query
    
    Returns
    -------
    df: `pandas.DataFrame`
        Dataframe containing the result of the query
    """
    from sqlalchemy.engine.base import Engine
    import pandas
    # If no connection is specified check in the local default directory
    if connection is None:
        ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
        connection = 'sqlite:///'+os.path.join(ROOT_DIR,'decam.db')
    # If the connection is a string instead of an engine, create a db Engine from the connection
    if not isinstance(connection, Engine):
        from sqlalchemy import create_engine
        connection = create_engine(connection)
    df = pandas.read_sql(sql, connection)
    return df