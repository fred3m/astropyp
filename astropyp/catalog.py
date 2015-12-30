import numpy as np
from collections import OrderedDict
import logging
import astropy.units as apu
from astropy import table
from astropy.extern import six
from astropy import coordinates

from astropyp.utils import misc

logger = logging.getLogger('astropyp.catalog')

class Catalog(object):
    """
    Wrapper for `~astropy.table.Table` catalogs of point sources.
    This allows users to map class attributes to columns in the
    source table, at times with different names, making it easier to
    work with catalogs of objects with different column names.
    
    Parameters
    ----------
    sources: `~astropy.table.Table`
        Astropy table that contains a list of point sources.
    use_defaults: bool, optional
        Whether or not to use the default column mapping
        (x, y, ra, dec, e_ra, e_dec, aper_flux, aper_flux_err,
        psf_flux, psf_flux_err). *Default=True*
    kwargs: Keyword Arguments
        Key,value pairs that assign static column names to
        columns in the ``sources`` table. This can override any
        names in defaults as well as add new column names.
    """
    def __init__(self, sources=None, use_defaults=True, **kwargs):
        default_columns = ['x','y','ra','dec','e_ra','e_dec',
            'aper_flux','aper_flux_err','psf_flux','psf_flux_err']
        
        self.sources = sources
        # If using default columns, update them with any keyword
        # arguments passed to init
        if use_defaults:
            self._columns = OrderedDict([(k,k) for k in default_columns])
            self._columns.update(kwargs)
        else:
            self._columns = kwargs
        # Add all of the static columns to the Catalog properties
        for col in self._columns.keys():
            setattr(self.__class__, col, self._add_property(col))
    
    @property
    def shape(self):
        return (len(self.sources), len(self.sources.columns))
    
    def _add_property(self, key):
        """
        Creating a mapping from a key using the current
        value in the lookup table
        """
        # Get the static column
        def getter(self):
            prop = self._columns[key]
            if prop in self.sources.columns:
                result = self.sources[prop]
            else:
                result = None
            return result
        # Set the static column in the table
        def setter(self, value):
            prop = self._columns[key]
            self.sources[prop] = value
        # Delete the static column from the table
        def deleter(self):
            prop = self._columns[key]
            if prop in self.sources.columns:
                del self.sources[prop]
        return property(getter, setter, deleter)
        
    @property
    def columns(self):
        """
        Columns in the ``sources`` table
        """
        return self.sources.columns
    @property
    def static_columns(self):
        """
        Return a list of static column names, but only the ones
        with a valid mapping to the ``sources`` table
        """
        columns = [k for k,v in self._columns.items() 
            if v in self.sources.columns]
        return columns
    
    def update_static_column(self, static_name, column_name):
        """
        Change the mapping from a static column to a column
        in the ``sources`` table
        """
        self._columns[static_name] = column_name
        setattr(self.__class__, static_name, 
            self._add_property(static_name))
    
    def __getitem__(self, arg):
        """
        Index the Catalog the same as indexing an
        astropy table
        """
        keys = arg
        for k,v in self._columns.items():
            if hasattr(arg, '__iter__'):
                keys = [v if key==k else key for key in keys]
            else:
                if k==arg:
                    keys = v
                    break
                
        return self.sources[keys]

def match_catalogs(ra0,dec0,ra1,dec1,units0='deg',units1='deg', 
        separation=1*apu.arcsec):
    c0 = coordinates.SkyCoord(ra0,dec0,unit=units0)
    c1 = coordinates.SkyCoord(ra1,dec1,unit=units1)
    idx1, idx0, d2, d3 = c0.search_around_sky(c1, separation)
    return idx0, idx1, d2

def match_all_catalogs(catalogs, ra_names, dec_names, units='deg', 
        separation=1*apu.arcsec, combine=True):
    """
    Match a list of catalogs based on their ra, dec, and separation
    with an inner join.
    """
    # If numbers are passed to the parameters, turn them into lists
    if isinstance(ra_names, six.string_types):
        ra_names = [ra_names for n in range(len(catalogs))]
    if isinstance(dec_names, six.string_types):
        dec_names = [dec_names for n in range(len(catalogs))]
    if isinstance(units, six.string_types):
        units = [units for n in range(len(catalogs))]
    
    for n in range(1, len(catalogs)):
        idx0, idx1, d2 = match_catalogs(
            catalogs[0][ra_names[0]], catalogs[0][dec_names[0]],
            catalogs[n][ra_names[n]], catalogs[n][dec_names[n]],
            units[0], units[n], separation)
        for m in range(n):
            catalogs[m] = catalogs[m][idx0]
        catalogs[n] = catalogs[n][idx1]
    if combine:
        from astropy.table import vstack
        catalogs = vstack(catalogs)
    return catalogs

def combine_coordinates(all_ra, all_dec, method='mean'):
    """
    Take the median, mean, or most accurate position of a set of catalogs
    """
    if method=='mean':
        ra = np.ma.mean(all_ra, axis=0)
        dec = np.ma.mean(all_dec, axis=0)
    elif method=='median':
        ra = np.ma.median(all_ra, axis=0)
        dec = np.ma.median(all_dec, axis=0)
    elif method=='in order':
        ra = np.ma.array(all_ra[0])
        dec = np.ma.array(all_dec[0])
        for n in range(1,len(all_ra)):
            ra[ra.mask] = all_ra[n][ra.mask]
            dec[dec.mask] = all_dec[n][dec.mask]
    else:
        raise Exception("Method '{0}' is currently not supported".format(method))
    return ra,dec

def merge_observations(ra0, dec0, ra1, dec1, units0='deg', units1='deg', 
        separation=1*apu.arcsec):
    """
    Outer join of two sets of observations
    
    Result
    ------
    idx0: masked array
        Indices to match obs0 with obs1
    idx1: masked array
        Indices to match obs1 with obs0
    d2: masked array
        Distance between observations
    matched: boolean array
        Observations that are matched in obs0 and obs1
    """
    init_idx0 = np.array([n for n in range(ra0.shape[0])])
    init_idx1 = np.array([n for n in range(ra1.shape[0])])
    idx0, idx1, d2 = match_catalogs(
        ra0,dec0,ra1,dec1,units0=units0,units1=units1, separation=separation)
    unmatched0 = np.delete(init_idx0, idx0)
    unmatched1 = np.delete(init_idx1, idx1)

    idx0 = np.hstack([idx0, unmatched0, -np.ones(unmatched1.shape, dtype=int)])
    idx1 = np.hstack([idx1, -np.ones(unmatched0.shape, dtype=int), unmatched1])
    new_d2 = np.hstack([
            np.array(d2), 
            np.ones(unmatched0.shape)*np.inf, 
            np.ones(unmatched1.shape)*np.inf])
    d2 = coordinates.Angle(new_d2, unit=d2.unit)    
    mask0 = idx0<0
    mask1 = idx1<0
    idx0 = np.ma.array(idx0, mask=mask0)
    idx1 = np.ma.array(idx1, mask=mask1)
    matched = ~(mask0 | mask1)
    return idx0, idx1, d2, matched

def get_merged_indices(all_ra, all_dec, separation=1*apu.arcsec):
    """
    Get masked indices for a set of ra,dec that merge the 
    sources together using an outer join
    """
    def avg_masked_arrays(arrays):
        return np.ma.mean(np.ma.vstack(arrays), axis=0)

    indices = [np.ma.array([n for n in range(all_ra[m].shape[0])], dtype=int) 
        for m in range(len(all_ra))]
    mean_ra = np.ma.array(all_ra[0])
    mean_dec = np.ma.array(all_dec[0])
    for n in range(1,len(all_ra)):
        idx1, idx0, d2, matched = merge_observations(
            np.ma.array(all_ra[n]), np.ma.array(all_dec[n]), mean_ra, mean_dec)
        mean_ra = misc.update_ma_idx(mean_ra,idx0)
        mean_dec = misc.update_ma_idx(mean_dec,idx0)
        new_ra = misc.update_ma_idx(all_ra[n],idx1)
        new_dec = misc.update_ma_idx(all_dec[n],idx1)
        mean_ra = np.ma.mean(np.ma.vstack([mean_ra, new_ra]), axis=0)
        mean_dec = np.ma.mean(np.ma.vstack([mean_dec, new_dec]), axis=0)
        for m in range(n):
            indices[m] = misc.update_ma_idx(indices[m],idx0)
        indices[n] = idx1
    matched = np.sum([i.mask for i in indices],axis=0)==0
    return indices, matched

def merge_catalogs(catalogs,
        fields=['ra','dec','psf_mag','peak'], idx_name='all_obs',
        ref_catalogs=None, combine_method='mean', cat_names=None, 
        refcat_names=None, refcat_fields=None, separation=1*apu.arcsec):
    all_ra = [cat.ra for cat in catalogs]
    all_dec = [cat.dec for cat in catalogs]
    
    if cat_names is None:
        cat_names = ['cat_{0}'.format(n) for n in range(len(catalogs))]
    
    indices, matched = get_merged_indices(all_ra, all_dec, separation)
    
    #return all_ra, all_dec, indices, matched
    
    # Match catalogs using their merged indices
    tbl = table.Table(masked=True)
    #matched_catalogs = [
    #    cat.sources[indices[n]] for n, cat in enumerate(catalogs)]
    all_ra = [all_ra[n][indices[n]] for n in range(len(catalogs))]
    all_dec = [all_dec[n][indices[n]] for n in range(len(catalogs))]
    tbl['ra'],tbl['dec'] = combine_coordinates(all_ra, all_dec, combine_method)
    
    #for n,sources in enumerate(matched_catalogs):
    for n,catalog in enumerate(catalogs):
        sources = catalog.sources
        cat_name = cat_names[n]
        for field in fields:
            if field in sources.columns.keys():
                tbl[cat_name+'_'+field] = misc.update_ma_idx(
                    np.array(sources[field]),indices[n])
    # Match reference catlogs
    if ref_catalogs is not None:
        if refcat_names is None:
            refcat_names = [
                'refcat_{0}'.format(n) for n in range(len(ref_catalogs))]
        if refcat_fields is None:
            raise Exception(
                "Expected a list of refcat fields to include in the catalog")
        
        for n, catalog in enumerate(ref_catalogs):
            if not hasattr(tbl['ra'], 'unit') or tbl['ra'].unit is None:
                src_unit = 'deg'
            else:
                src_unit = tbl['ra'].unit
            if not hasattr(catalog.ra, 'unit') or catalog.ra.unit is None:
                cat_unit = 'deg'
            else:
                cat_unit = catalog.ra.unit
            # Unlike source catalog, reference catalogs only include the rows 
            # that have matches in the
            # main source catalog (tbl)
            coords = coordinates.SkyCoord(tbl['ra'], tbl['dec'], unit=src_unit)
            cat_coords = coordinates.SkyCoord(
                catalog.ra, catalog.dec, unit=cat_unit)
            idx, d2, d3 = coords.match_to_catalog_sky(cat_coords)
            idx = np.ma.array(idx)
            idx.mask = d2>separation
            # Add columns from reference catalogs
            for field in refcat_fields:
                if field in catalog.columns.keys():
                    tbl[refcat_names[n]+'_'+field] = misc.update_ma_idx(
                        np.array(catalog[field]),idx)
    return tbl

def save_catalog(tbl, connection, tbl_name, frame, if_exists='append'):
    """
    Save sources to source list to a database. 
    This will overwrite any sources saved from the same frame, 
    but sources from other frames in the same table will be 
    unaffected. *This function requires SQLAlchemy*
    
    Parameters
    ----------
    tbl: `~astropy.table/Table` or `Catalog`
        Source catalog to save
    connection: string or `~sqlalchemy.engine.base.Engine`
        Either connection string or SQLAlchemy database engine
        to connect to the database
    tbl_name: string
        Name of the table to create or update in the database
    frame: int
        Name/Number of the frame (CCD in detector array) used
        to generate the catalog
    """
    from astropyp import db_utils
    # Connect to the database
    engine = db_utils.index.init_connection(connection)
    meta, db_tbl, session = db_utils.index.connect2idx(engine, 
        tbl_name)
    
    # First delete all of the rows for the current frame (if any exist)
    if db_tbl is not None:
        engine.execute(db_tbl.delete().where(db_tbl.c.frame==frame))
    # Append the current source list to the list of sources from other CCDs
    if isinstance(tbl, Catalog):
        tbl = tbl.sources
    tbl['frame'] = frame#np.ones((len(tbl),),dtype=int)*frame
    df = tbl.to_pandas()
    df.to_sql(tbl_name, engine, if_exists=if_exists)