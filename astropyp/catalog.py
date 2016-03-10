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
    
    def __len__(self):
        return len(self.sources)

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
        raise Exception(
            "Method '{0}' is currently not supported".format(method))
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

def old_get_merged_indices(all_ra, all_dec, separation=1*apu.arcsec):
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
    
    indices, matched = old_get_merged_indices(all_ra, all_dec, separation)
    
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

def find_neighbors(radius, positions=None, kd_tree=None):
    """
    Find all neighbors within radius of each source in a list of
    positions or a KD-Tree.
    
    Parameters
    ----------
    radius: float
        Maximum distance for a neighbor (center to center) to
        be included
    positions: array or list of tuples, optional
        Array or list of coord1,coord2 positions to use for
        neighbor search. If ``positions`` is not specified,
        kd_tree must be given.
    kd_tree: `~scipy.spatial.cKDTree`
        KD Tree to use for the search. If this isn't specified
        then a list of positions must be specified
    
    Result
    ------
    idx: np.array
        List of indices of all sources with a neighbor. Sources with
        multiple neighbors will have multiple entries, one for each neighbor
        given in ``nidx``.
    nidx: np.array
        List of indices for neighbors matching each index in ``idx``.
    """
    from scipy import spatial
    if kd_tree is None:
        if positions is not None:
            KDTree = spatial.cKDTree
            kd_tree = KDTree(positions)
        else:
            raise Exception("You must either specify a list "
                "of positions or a kd_tree")
    pairs = kd_tree.query_pairs(radius)
    neighbors = np.array(list(pairs))
    if len(neighbors)==0:
        return np.array([], dtype=int), np.array([], dtype=int)
    neighbors = np.vstack([neighbors,np.fliplr(neighbors)])
    idx = neighbors[:,0]
    nidx = neighbors[:,1]
    sort_idx = np.argsort(idx)
    return idx[sort_idx], nidx[sort_idx]

def get_merged_indices(coords, ref_coords=None, kdtree=None, 
        pool_size=None, separation=1/3600):
    """
    Get the indices to merge two sets of coordinates together. This includes
    the incides to match both sets, indices of the unmatched rows, and 
    indices of rows that had multiple matches.
    
    Parameters
    ----------
    coords: array-like
        A 2D array with either 2 columns of coordinates (coord1, coord2),
        where coord1 and coord2 are Nx1 arrays, or
        N rows of coordinate pairs 
        [(coord1_1, coord1_2),(coord2_1,coord2_2),...]
        where coordN_1 and coordN_2 are floats.
    ref_coords: array-like, optional
        A 2D array with either 2 columns of coordinates or N rows of
        coordinate pairs (see ``coords`` for more). 
        Either ``ref_coords`` or ``kdtree`` must be specified.
    kdtree: `spatial.cKDTREE`, optional
        KDTree of the reference coordinates (this is an object to use
        for matching positions of objects in k-dimensions, in 2 dimensions
        it is a quad-tree).
        Either ``ref_coords`` or ``kdtree`` must be specified.
    pool_size: int, optional
        Number of processors to use to match coordinates. If 
        ``pool_size is None`` (default) then the maximum number
        of processors is used.
    separation: float, optional
        Maximum distance between two coordinates for a match. The default
        is ``1/3600``, or 1 arcsec.
    
    Returns
    -------
    ref_indices: tuple(idx1, unmatched1)
        Indices to match the reference coordinates to the observed
        coordinates. So the rows of ref_coords[idx1]~coords[idx2],
        ref_coords[unmatched1]~coords[unmatched2], if
        ref_coords and coords are Nx2 arrays of coordinate pairs;
    coord_indices: tuple(idx2, unmatched2)
        Indices to match the coordinates to the reference coordinates.
    duplicates: array-like
        Indices of coordinates with multiple matches, so that
        ref_coords[idx1][duplicates]~coords[idx2][duplicates]
    """
    # If the user didn't specify a KDTREE, 
    if kdtree is None:
        try:
            from scipy import spatial
        except ImportError:
            raise ImportError(
                "You must have 'scipy' installed to combine catalogs")
        if ref_coords is not None:
            if len(ref_coords)==2:
                ref1,ref2 = ref_coords
                pos1 = np.array([ref1,ref2])
                pos1 = pos1.T
            elif len(ref_coords[0])==2:
                pos1 = ref_coords
            else:
                raise ValueError(
                    "Expected either a 2xN array or Nx2 array for ref_coords")
            KDTree = spatial.cKDTree
            kdtree = KDTree(pos1)
        else:
            raise Exception("Either ref_coords or kdtree must be specified")
    if pool_size is None:
        pool_size = -1
    
    if len(coords)==2:
        coord1, coord2 = coords
        pos2 = np.array([coord1,coord2])
        pos2 = pos2.T
    elif len(coords[0])==2:
        pos2 = coords
    else:
        raise ValueError("Expected either a 2xN array or Nx2 array for coords")
    
    src_count1 = len(ref1)
    src_count2 = len(coord1)
    
    # Match all of the sources
    d2,idx = kdtree.query(pos2, n_jobs=pool_size,
        distance_upper_bound=separation)
    matches = np.isfinite(d2)
    idx1 = idx[matches]
    idx2 = np.where(matches)[0]
    # Flag all of the duplicate sources
    unique, inverse, counts = np.unique(
        idx1, return_inverse=True, return_counts=True)
    u = unique.copy()
    cidx = counts>1
    u[cidx]=-1
    didx = u[inverse]<0
    duplicates = np.arange(len(idx1))[didx]
    # Identify unmatched sources
    unmatched1 = np.setdiff1d(range(src_count1), idx1)
    unmatched2 = np.setdiff1d(range(src_count2), idx2)
    
    #return (idx1, unmatched1, duplicate1), (idx2, unmatched2, duplicate2)
    return (idx2, unmatched2),(idx1, unmatched1), duplicates

def get_all_merged_indices(all_coord1, all_coord2, pool_size=None, 
        separation=1/3600., merge_type='outer'):
    """
    Get masked indices for a set of ra,dec that merge the 
    sources together using an outer join
    
    Parameters
    ----------
    all_coord1: list of array-like
        List of arrays of values for the first coordinate (usually RA or X)
    all_coord2: list of array-like
        List of arrays of values for the second coordinate (usually DEC or Y)
    pool_size: int, optional
        Number of processors to use to match coordinates. If 
        ``pool_size is None`` (default) then the maximum number
        of processors is used.
    separation: float, optional
        Maximum distance between two coordinates for a match. The default
        is ``1/3600``, or 1 arcsec.
    merge_type: str, optional
        Type of merge to use. This must be 'outer','inner', 'left', or 'right'.
        The default is 'outer'.
    
    Returns
    -------
    indices: list of masked arrays
        Indices to match each catalog in all_coord1, all_coord2 to the
        master catalog
    matched: array
        Indices of rows that has an entry for *every* set of coordinates
    all_duplicates: array
        Indices of rows that have duplicate values
    """
    from astropyp.utils import misc
    
    if merge_type not in ['outer','inner','left','right']:
        raise ValueError(
            "merge_type must be 'outer','inner', 'left', or 'right'")
    
    # Initialize indices and coordinates
    indices = [np.ma.array([n for n in 
        range(all_coord1[m].shape[0])], dtype=int) 
            for m in range(len(all_coord1))]
    mean_coord1 = np.ma.array(all_coord1[0])
    mean_coord2 = np.ma.array(all_coord2[0])
    all_duplicates = np.zeros(mean_coord1.shape, dtype=bool)
    
    # Create merged indices
    for n in range(1,len(all_coord1)):
        idx0, idx1, duplicates = get_merged_indices(
            (all_coord1[n],all_coord2[n]), 
            ref_coords=(mean_coord1,mean_coord2), 
            pool_size=pool_size, separation=separation)
        new_idx, new_unmatched = idx0
        ref_idx, ref_unmatched = idx1
        
        # Update list of duplicates
        duplicates_unmatched = all_duplicates[ref_unmatched]
        all_duplicates = all_duplicates[ref_idx]
        all_duplicates[duplicates] = True
        
        # Update indices
        if merge_type=='outer' or merge_type=='left':
            ref_idx = np.hstack([ref_idx, ref_unmatched])
            new_idx = np.hstack([new_idx, 
                -np.ones(ref_unmatched.shape, dtype=int)])
            all_duplicates = np.hstack([all_duplicates, duplicates_unmatched])
        if merge_type=='outer' or merge_type=='right':
            ref_idx = np.hstack([ref_idx, 
                -np.ones(new_unmatched.shape, dtype=int)])
            new_idx = np.hstack([new_idx, new_unmatched])
            all_duplicates = np.hstack([all_duplicates, 
                np.zeros(new_unmatched.shape, dtype=bool)])
        # Mask indices
        ref_mask = ref_idx<0
        new_mask = new_idx<0
        ref_idx = np.ma.array(ref_idx, mask=ref_mask)
        new_idx = np.ma.array(new_idx, mask=new_mask)
        
        # Update the mean coordinate values
        mean_coord1 = misc.update_ma_idx(mean_coord1,ref_idx)
        mean_coord2 = misc.update_ma_idx(mean_coord2,ref_idx)
        new_coord1 = misc.update_ma_idx(all_coord1[n],new_idx)
        new_coord2 = misc.update_ma_idx(all_coord2[n],new_idx)
        mean_coord1 = np.ma.mean(
            np.ma.vstack([mean_coord1, new_coord1]), axis=0)
        mean_coord2 = np.ma.mean(
            np.ma.vstack([mean_coord2, new_coord2]), axis=0)
        # Update all of the indices with the new matches
        for m in range(n):
            indices[m] = misc.update_ma_idx(indices[m],ref_idx)
        indices[n] = new_idx
        
    matched = np.sum([i.mask for i in indices],axis=0)==0
    return indices, matched, all_duplicates

def mask_catalog_columns(catalog, idx, columns=None, 
        catname=None, new_columns=None,
        catalog_kwargs=None):
    """
    Mask all of the rows in a table (or subset of columns in a table) 
    with a masked index and (optionally) rename the columns.
    
    Parameters
    ----------
    catalog: `~astropy.table.Table` or `~Catalog`
        Catalog or Table to be masked
    idx: `~numpy.ma.array`
        Masked array of indices to use for updating the catalog
    columns: list of strings, optional
        Columns to include in the masked table. If ``columns is None`` 
        (default) then all of the columns are included in the masked catalog.
    catname: str, optional
        Name of catalog. This is only necessary if you with to rename the
        columns of the catalog for stacking later. See ``new_columns`` 
        for more.
    new_columns: list of strings, optional
        New names for the columns. This may be useful if you are combining 
        catalogs and want to standardize the column names. If
        ``new_columns is None`` (default) then if a ``catname`` is provided all
        of the columns are renamed to 'columnname_catname', otherwise 
        the original column names are used.
    catalog_kwargs: dict
        If the result is desired to be a `~astropyp.catalog.Catalog` then
        these are the kwargs to specify when initializing the catalog
        (for example names of the ra,dec,x,y columns). Otherwise
        an `~astropy.table.Table` is returned
    
    Returns
    -------
    tbl: `~astropy.table.Table` or `~astropyp.catalog.Catalog`
        Catalog created by applying the masked index. The type of object
        returned depends on if ``catalog_kwargs is None`` (default), which
        returns a Table. Otherwise a Catalog is returned.
    """
    from astropy.table import Table
    if isinstance(catalog, Catalog):
        tbl = catalog.sources
    else:
        tbl = catalog
    new_tbl = Table(masked=True)
    if columns is None:
        columns = tbl.columns.keys()
    if new_columns is None:
        if catname is None:
            new_columns = columns
        else:
            new_columns = ['{0}_{1}'.format(col,catname) for col in columns]
    for n, col in enumerate(columns):
        new_tbl[new_columns[n]] = misc.update_ma_idx(tbl[col], idx)
    if catalog_kwargs is not None:
        new_tbl = Catalog(new_tbl, **catalog_kwargs)
    return new_tbl