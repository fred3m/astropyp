import logging
logger = logging.getLogger('astropyp.plots.phot')

def build_rough_coeff_plot(sources, mag_name, ref_name):
    """
    Sometimes it is helpful to plot the average diffence for a given FOV or CCD.
    This can help identify observations that were not photometric as the
    avg difference vs airmass should be roughly linear.
    
    Parameters
    ----------
    sources: `astropy.table.Table`
        Catalog of observations
    mag_name: str
        Name of the magnitude column in ``sources``
    ref_name: str
        Name of the reference magnitude column in ``sources
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    x = []
    y = []
    err = []
    for airmass in np.unique(sources['airmass']).tolist():
        x.append(airmass)
        obs = sources[sources['airmass']==airmass]
        obs = obs[np.isfinite(obs[mag_name]) & np.isfinite(obs[ref_name])]
        diff = obs[mag_name]-obs[ref_name]
        obs = obs[np.sqrt((diff-np.mean(diff))**2)<np.std(diff)]
        median = np.median(obs[mag_name]-obs[ref_name])
        y.append(median)
        err.append(np.std(obs[mag_name]-obs[ref_name]))
    plt.xlabel('Airmass')
    plt.ylabel('Median difference in {0} magnitude'.format(mag_name))
    plt.plot(x,y)
    plt.errorbar(x, y, yerr=err, fmt='-o')
    plt.show()

def build_diff_plot(obs, mag_name, ref_name, mag_err_name, ref_err_name, 
        plot_format='b.', show_stats=True, clipping=1, filename=None, title=None):
    """
    Plot the difference between reference magnitudes and observed magnitudes for a 
    given set of observations
    
    Parameters
    ----------
    obs: `astropy.table.Table`
        Catalog of observations to compare
    mag_name: str
        Name of the magniude column in ``obs`` to compare
    ref_name: str
        Name of the reference column in ``obs`` to compare
    mag_err_name: str
        Name of the magnitude error column
    ref_err_name: str
        Name of the reference error column
    plot_format: str
        Format for matplotlib plot points
    show_stats: str
        Whether or not to show the mean and standard deviation of the observations
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    
    diff = obs[mag_name]-obs[ref_name]
    # Remove outlier sources
    plot_obs = obs[np.sqrt((diff-np.mean(diff))**2) < clipping*np.std(diff)]
    # build plot
    x = plot_obs[mag_name]
    y = plot_obs[mag_name]-plot_obs[ref_name]
    err = np.sqrt(plot_obs[mag_err_name]**2+plot_obs[ref_err_name]**2)
    plt.errorbar(x, y, yerr=err, fmt=plot_format)
    plt.xlabel(mag_name)
    plt.ylabel('Diff from Std Sources')
    if title is not None:
        plt.title(title)
    # show stats
    if show_stats:
        logger.info('mean: {0}'.format(np.mean(y)))
        logger.info('std dev: {0}'.format(np.std(y)))
    # Save plot or plot to screen
    if filename is None:
        plt.show()
    else:
        plt.save(filename)
    plt.close()
    return y