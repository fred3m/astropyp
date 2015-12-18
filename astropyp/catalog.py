import numpy as np
from collections import OrderedDict
import logging
import astropy.units as apu
from astropy import table
from astropy.extern import six

logger = logging.getLogger('astropyp.catalog')

class ImageCatalog(object):
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
        # Add all of the static columns to the ImageCatalog properties
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
        Index the ImageCatalog the same as indexing an
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
    c0 = SkyCoord(ra0,dec0,unit=units0)
    c1 = SkyCoord(ra1,dec1,unit=units1)
    idx1, idx0, d2, d3 = c0.search_around_sky(c1, separation)
    #print len(c0), max(idx0)
    #print len(c1), max(idx1)
    return idx0, idx1, d2

def match_all_catalogs(catalogs, ra_names, dec_names, units='deg', 
        separation=1*apu.arcsec, combine=True):
    """
    Match a list of catalogs based on their ra, dec, and separation
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

def merge_catalogs(all_ra, all_dec, separation=1*apu.arcsec):
    def avg_masked_arrays(arrays):
        return np.ma.mean(np.ma.vstack(arrays), axis=0)

    indices = [np.ma.array([n for n in range(all_ra[m].shape[0])], dtype=int) 
        for m in range(len(all_ra))]
    mean_ra = np.ma.array(all_ra[0])
    mean_dec = np.ma.array(all_dec[0])
    for n in range(1,len(all_ra)):
        #print '\nn', n
        idx1, idx0, d2, matched = merge_observations(
            np.ma.array(all_ra[n]), np.ma.array(all_dec[n]), mean_ra, mean_dec)
        #print 'idx', idx0, idx1
        #print 'matched', matched
        mean_ra = update_array(mean_ra,idx0)
        mean_dec = update_array(mean_dec,idx0)
        new_ra = update_array(all_ra[n],idx1)
        new_dec = update_array(all_dec[n],idx1)
        #print 'mean before', mean_ra, mean_dec
        #print 'new before', new_ra, new_dec
        mean_ra = np.ma.mean(np.ma.vstack([mean_ra, new_ra]), axis=0)
        mean_dec = np.ma.mean(np.ma.vstack([mean_dec, new_dec]), axis=0)
        
        #print 'mean', mean_ra, mean_dec
        
        for m in range(n):
            #print 'm', m, indices[m]
            indices[m] = update_array(indices[m],idx0)
            #print 'm after', m, indices[m]
        indices[n] = idx1
    matched = np.sum([i.mask for i in indices],axis=0)==0
    return indices, matched