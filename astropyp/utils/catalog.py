import astropy.units as apu
from astropy.extern import six

def match_catalogs(ra0,dec0,ra1,dec1,units0='deg',units1='deg', separation=1*apu.arcsec):
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

    indices = [np.ma.array([n for n in range(all_ra[m].shape[0])], dtype=int) for m in range(len(all_ra))]
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