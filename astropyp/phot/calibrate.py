import numpy as np
from astropy.table import Table, join

def clip_color_outliers(color, ref_color, dbscan_kwargs={}, show_plots=True):
    """
    Use DBSCAN clustering to only select the points in color-color space
    that are part of the main cluster
    
    Parameters
    ----------
    color: array-like
        Array of instrumental colors
    ref_color: array-like
        Array of colors from a reference catalog
    dbscan_kwargs: dict
        Dictionary of keyword arguments to pass to DBSCAN. Typical 
        parameters are 'eps', the maximum separation for points in a
        group and 'min_samples', the minimum number of points for a
        cluster to be grouped together. No kwargs are required.
    
    Returns
    -------
    idx: array
        Indices of points that are NOT outliers
    groups: array
        Group numbers. Outliers have group number ``-1``.
    labels: array
        Label for each point in color, ref_color with the group
        number that it is a member of
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("You must have sklearn installed to clip outliers")
    
    coords = np.array([color, ref_color])
    coords = coords.T
    db = DBSCAN(**dbscan_kwargs).fit(coords)
    groups = np.unique(db.labels_)
    if show_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        #for gid in groups:
        #    idx = gid==db.labels_
        #    plt.plot(color[idx],ref_color[idx], '.')
        idx = db.labels_>=0
        plt.plot(color[~idx], ref_color[~idx], 'r.')
        plt.plot(color[idx], ref_color[idx], '.')
        plt.title('Outliers in red')
        plt.xlabel('instrumental color')
        plt.ylabel('reference color')
        plt.show()

    idx = db.labels_>-1
    return idx, groups, db.labels_

def calibrate_color(instr_color, airmass, a, b, k1, k2):
    """
    Transform colors to a different photometric system.
    See Landolt 2007 for more.
    """
    return a+b*(1/(1+k2*airmass))*(instr_color-k1*airmass)

def calculate_color_coeffs(instr_color, ref_color, airmass, init_params):
    """
    Using the procedure in Landolt 2007 we adjust the colors to a set
    of reference colors.
    """
    from scipy.optimize import curve_fit

    def get_color(measurements, a, b, k1, k2):
        # Unpack the observations and call the calibrate_color function
        # (for consistency)
        instr_color, airmass = measurements
        return calibrate_color(instr_color, airmass, a, b, k1, k2)
    
    results = curve_fit(get_color, [instr_color,airmass], ref_color, 
        init_params)
    return results

def calibrate_magnitude(instr_mag, airmass, ref_color, 
        zero, extinct, color, instr=None):
    """
    Calibrate instrumental magnitude to a standard photometric system.
    This assumes that ref_color has already been adjusted for any
    non-linearities between the instrumental magnitudes and the 
    photometric system. Otherwise use `~calibrate_magnitude_full`
    """
    if instr is None:
        result = instr_mag-extinct*airmass+zero+color*ref_color
    else:
        result = instr*(instr_mag-extinct*airmass)+zero+color*ref_color
    return result

def calibrate_magnitude_full(instr_mag, airmass, instr_color, 
        a, b, k1, k2, zero, extinct, color, instr=None):
    """
    Using the transformation in Landolt 2007, calibrate an instrumental
    magnitude to a photometric standard.
    """
    adjusted_color = calibrate_color(instr_color, airmass, a, b, k1, k2)
    result = calibrate_magnitude(instr_mag, airmass, adjusted_color, 
        zero, extinct, color, instr)
    return result

def calculate_izY_coeffs(instr_mag, instr_color, airmass,
        ref_mag, ref_color, dbscan_kwargs={}, show_plots=True,
        color_init_params=None, mag_init_params=None, cluster=True):
    """
    Use the procedure from Landolt 2007 to calibrate to a given 
    standard catalog (including adjusting the instrumental colors to the 
    standard catalog colors).
    """
    from scipy.optimize import curve_fit
    
    def get_mag(measurements, zero, extinct, color, instr=None):
        instr_mag, airmass, ref_color = measurements
        result = calibrate_magnitude(instr_mag, airmass, ref_color,
            zero, extinct, color, instr)
        return result
    
    # Calculate the coefficients to adjust to the standard colors
    if color_init_params is None:
        color_init_params = [2.,1.,.1,.1]
    
    if cluster:
        idx, groups, labels = clip_color_outliers(
            instr_color, ref_color, dbscan_kwargs, show_plots)
        color_result = calculate_color_coeffs(
            instr_color[idx], ref_color[idx], airmass[idx], color_init_params)
    else:
        color_result = calculate_color_coeffs(
            instr_color, ref_color, airmass, color_init_params)
    a,b,k1,k2 = color_result[0]
    adjusted_color = calibrate_color(instr_color, airmass, a, b, k1, k2)
    
    if show_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        for am in np.unique(airmass):
            aidx = airmass[idx]==am
            x = np.linspace(np.min(instr_color[idx][aidx]), 
                np.max(instr_color[idx][aidx]),10)
            y = calibrate_color(x, am, a, b, k1, k2)
            plt.plot(instr_color[idx][aidx], ref_color[idx][aidx], 
                '.', alpha=.1)
            plt.plot(x,y, 'r')
            plt.title('Airmass={0:.2f}'.format(am))
            plt.xlabel('Instrumental Color')
            plt.ylabel('Reference Color')
            plt.show()
        plt.plot(adjusted_color, ref_color, '.', alpha=.1)
        plt.title('Calibrated Color')
        plt.xlabel('Adjusted Color')
        plt.ylabel('Standard Color')
        plt.axis('equal')
        plt.show()
    
    # Fit the coefficients
    measurements = [instr_mag, airmass, adjusted_color]
    if mag_init_params is None:
        mag_init_params = [25.,.1,.1]
    mag_result = curve_fit(get_mag, measurements, ref_mag, mag_init_params)
    # Package the results
    if len(mag_result[0])==3:
        zero, extinct, color = mag_result[0]
        instr = None
    else:
        zero, extinct, color, instr = mag_result[0]
    results = color_result[0].tolist()+mag_result[0].tolist()
    
    if show_plots:
        mag = calibrate_magnitude_full(instr_mag, airmass, instr_color, 
            a, b, k1, k2, zero, extinct, color, instr)
        diff = mag-ref_mag
        rms = np.sqrt(np.mean(diff)**2+np.std(diff)**2)
        plt.plot(mag, diff, '.',alpha=.1)
        plt.title('Calibrated magnitudes, rms={0:.4f}'.format(rms))
        plt.ylim([-.15,.15])
        plt.xlabel('mag')
        plt.ylabel('diff from standard')
        plt.show()
    return results, color_result[1], mag_result[1]

def calculate_coeffs_by_frame(instr_mag, instr_color, airmass, 
        ref_mag, ref_color, catalog_frame, frames, 
        dbscan_kwargs={}, color_init_params=None, 
        mag_init_params=[25.,.1,.1], show_plots=True):
    """
    Calculate coefficients to transform instrumental magnitudes
    to a standard photometric catalog individually for each frame.
    """
    # Clip outliers from the entire catalog
    idx, groups, labels = clip_color_outliers(
        instr_color, ref_color, dbscan_kwargs, show_plots)
    mag = np.zeros((np.sum(idx),))
    mag[:] = np.nan
    
    # Create a table to hold the coefficients
    frame_count = len(frames)
    a = np.zeros((frame_count,), dtype=float)
    b = np.zeros((frame_count,), dtype=float)
    k1 = np.zeros((frame_count,), dtype=float)
    k2 = np.zeros((frame_count,), dtype=float)
    zero = np.zeros((frame_count,), dtype=float)
    color = np.zeros((frame_count,), dtype=float)
    extinct = np.zeros((frame_count,), dtype=float)
    instr = np.zeros((frame_count,), dtype=float)
    frame_coeff = np.zeros((frame_count,), dtype='S4')
    
    # For each frame calculate the coefficients
    for n,frame in enumerate(frames):
        fidx = catalog_frame[idx]==frame
        result = calculate_izY_coeffs(
            instr_mag[idx][fidx], instr_color[idx][fidx], airmass[idx][fidx], 
            ref_mag[idx][fidx], ref_color[idx][fidx],
            dbscan_kwargs, show_plots=False, cluster=False,
            color_init_params=color_init_params,
            mag_init_params=mag_init_params)
        if len(mag_init_params)==3:
            a[n],b[n],k1[n],k2[n],zero[n],extinct[n],color[n] = result[0]
        else:
            a[n],b[n],k1[n],k2[n],zero[n],extinct[n],color[n],instr[n]=result[0]
        frame_coeff[n] = frame
    # Build the table
    if len(mag_init_params)==3:
        result = Table([a,b,k1,k2,zero,extinct,color,frame_coeff], 
            names=('a','b','k1','k2','zero','extinct','color','frame'))
    else:
        result = Table([a,b,k1,k2,zero,extinct,color,instr,frame_coeff], 
            names=('a','b','k1','k2','zero','extinct','color','instr','frame'))
    return result

def calibrate_photometry_by_frame(instr_mag, instr_color, airmass, 
        catalog_frame, coeffs):
    """
    Transform instrumental magnitudes to a standard photometric 
    catalog using different coefficients for each frame
    """
    catalog = Table([catalog_frame, np.arange(len(instr_mag), dtype=int)], 
        names=('frame','index'))
    joint_tbl = join(catalog, coeffs)
    joint_tbl.sort('index')
    if 'instr' in coeffs.columns.keys():
        mag = calibrate_magnitude_full(
            instr_mag, airmass, instr_color,
            joint_tbl['a'], joint_tbl['b'], joint_tbl['k1'], 
            joint_tbl['k2'], joint_tbl['zero'], 
            joint_tbl['extinct'], joint_tbl['color'],
            joint_tbl['instr'])
    else:
        mag = calibrate_magnitude_full(
            instr_mag, airmass, instr_color,
            joint_tbl['a'], joint_tbl['b'], joint_tbl['k1'], 
            joint_tbl['k2'], joint_tbl['zero'], 
            joint_tbl['extinct'], joint_tbl['color'])
    return mag