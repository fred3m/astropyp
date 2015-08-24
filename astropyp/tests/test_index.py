from astropyp import index
import os

def data_path(filename=''):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, filename)

def test_add_tbl(tmpdir):
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy import Integer, Float, String
    
    connection = 'sqlite:///'+str(tmpdir)+'test.db'
    columns = [
        ('id', Integer, {'primary_key':True}),
        ('test1', String, {'index': True}),
        ('test2', Float, {}),
    ]
    
    result = index.add_tbl(connection, tbl_name='obs', columns=columns)
    assert result == False
    
    # Test table was created
    meta = MetaData()
    engine = create_engine(connection)
    meta.reflect(engine)
    assert 'obs' in meta.tables.keys()
    
    # Test duplicate table
    result = index.add_tbl(connection, tbl_name='obs', columns=columns)
    assert result == True
    
def test_add_files_and_query(tmpdir):
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy import Integer, Float, String
    
    connection = 'sqlite:///'+str(tmpdir)+'test.db'
    columns = [
        ('id', Integer, {'primary_key':True}),
        ('EXPNUM', Integer, {'index': True}),
        ('FILETYPE', String, {}),
        ('IMGTYPE', String, {}),
        ('RA', Float, {}),
        ('DEC', Float, {}),
        ('EXPTIME', Float, {}),
        ('MJD', Float, {}),
        ('dummy', Integer, {}),
        ('filename', String, {})
    ]
    
    index.add_tbl(connection, columns, 'obs')
    new_files, duplicates = index.add_files(connection, 'obs', paths=data_path())
    tbl = index.query('select * from obs', connection)
    # Check that all files were saved
    assert len(tbl)==10
    assert len(new_files)==10
    assert len(duplicates)==0
    # Check that the dummy column loaded but is None
    assert tbl[0]['dummy'] is None
    
    # Check that duplicates are not re-written
    new_files, duplicates = index.add_files(connection, 'obs', paths=data_path())
    tbl = index.query('select * from obs', connection)
    assert len(tbl)==10
    assert len(new_files)==0
    assert len(duplicates)==10
    
    tbl = index.query('select * from obs where MJD<1', connection)
    assert len(tbl)==0

def test_filename_duplicates(tmpdir):
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy import Integer, Float, String
    
    connection = 'sqlite:///'+str(tmpdir)+'test.db'
    columns = [
        ('id', Integer, {'primary_key':True}),
        ('EXPNUM', Integer, {'index': True}),
        ('FILETYPE', String, {}),
        ('IMGTYPE', String, {}),
        ('RA', Float, {}),
        ('DEC', Float, {}),
        ('EXPTIME', Float, {}),
        ('MJD', Float, {}),
        ('dummy', Integer, {}),
        ('filename', String, {})
    ]
    filenames = data_path('fake_obs0.fits')
    
    index.add_tbl(connection, columns, 'obs')
    new_files, duplicates = index.add_files(connection, 
        'obs', filenames=filenames, paths=data_path())
    tbl = index.query('select * from obs', connection)
    
    assert len(tbl)==10
    assert len(new_files)==10
    assert len(duplicates)==0
    
    # Check that duplicates are not re-written
    new_files, duplicates = index.add_files(connection, 
        'obs', filenames=filenames, paths=data_path())
    tbl = index.query('select * from obs', connection)
    assert len(tbl)==10
    assert len(new_files)==0
    assert len(duplicates)==10

def test_get_distinct(tmpdir):
    from sqlalchemy import create_engine, MetaData
    from sqlalchemy import Integer, Float, String
    
    connection = 'sqlite:///'+str(tmpdir)+'test.db'
    columns = [
        ('id', Integer, {'primary_key':True}),
        ('EXPNUM', Integer, {'index': True}),
        ('FILETYPE', String, {}),
        ('IMGTYPE', String, {}),
        ('RA', Float, {}),
        ('DEC', Float, {}),
        ('EXPTIME', Float, {}),
        ('MJD', Float, {}),
        ('dummy', Integer, {}),
        ('filename', String, {})
    ]
    filenames = data_path('fake_obs0.fits')
    
    index.add_tbl(connection, columns, 'obs')
    new_files, duplicates = index.add_files(connection, 
        'obs', filenames=filenames, paths=data_path())
    distinct = index.get_distinct(connection, 'obs', 'EXPNUM')
    assert distinct==[None,0,1,2]