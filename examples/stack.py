from __future__ import division, print_function
import numpy as np
from collections import OrderedDict
import logging

import matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.wcs
from astropy import coordinates
import astropy.units as apu
from astropy import table

# Astropy gives a lot of obnxious warnings that
# are difficult to filter individually
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

import astropyp
from astropyp.wrappers.astromatic import ldac
from astropyp.phot import stack
import bd_search

#import cPickle as pickle
import dill as pickle

alogger = logging.getLogger('astropyp')
alogger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
alogger.addHandler(ch)

def get_exp_files(expnum, night, filtr, idx_connect):
    sql = 'select * from decam_obs where expnum={0}'.format(expnum)
    sql += ' and filter like "{0}%" and dtcaldat="{1}"'.format(filtr, night)
    exp_info = astropyp.db_utils.index.query(sql, idx_connect)
    img_filename = exp_info[exp_info['PRODTYPE']=='image'][0]['filename']
    img = fits.open(img_filename)
    dqmask_filename = exp_info[exp_info['PRODTYPE']=='dqmask'][0]['filename']
    dqmask = fits.open(dqmask_filename)
    return img, dqmask

idx_connect = 'sqlite:////media/data-beta/users/fmooleka/decam/decam.db'
ref_path = '/media/data-beta/users/fmooleka/decam/catalogs/ref/'

# SExtractor 'extract' detection parameters
conv_filter = np.load('/media/data-beta/users/fmooleka/2016decam/5x5gauss.npy')
sex_params = {
    'extract': {
        'thresh': 40,
        #'err':,
        'minarea': 3, # default
        'conv': conv_filter,
        #'deblend_nthresh': 32, #default
        'deblend_cont': 0.001,
        #'clean': True, #default
        #'clean_param': 1 #default
    },
    'kron_k': 2.5,
    'kron_min_radius': 3.5,
    'filter': conv_filter,
    #'thresh': 1.5 # *bkg.globalrms
}

obj='F100'
refname = '2MASS'
#refname = 'UCAC4'
fullref = ldac.get_table_from_ldac(ref_path+'{0}-{1}.fits'.format(obj,refname))

min_flux = 1000
min_amplitude = 1000
good_amplitude = 50
calibrate_amplitude = 200
frame = 1
#explist = [442430, 442431, 442432]
explist = [442433, 442434, 442435]
aper_radius = 8

ccds = []

# Open the files
for expnum in explist:
    img, dqmask = get_exp_files(expnum, "2015-05-26", "z", idx_connect)
    header = img[frame].header
    wcs = astropy.wcs.WCS(header)
    img_data = img[frame].data
    dqmask_data = dqmask[frame].data
    ccd = astropyp.phot.phot.SingleImage(header, img_data, dqmask_data,
        wcs=wcs, gain=4., exptime=30, aper_radius=aper_radius)
    ccds.append(ccd)

# Detect sources
ccd_stack = stack.Stack(ccds, 1)
ccd_stack.detect_sources(min_flux=min_flux, good_amplitude=good_amplitude,
    calibrate_amplitude=calibrate_amplitude, psf_amplitude=1000, 
    sex_params=sex_params, subtract_bkg=True, windowed=False)
# Get astrometric solution
ccd_stack.get_transforms()
# Create a catalog with all of the sources found on all of the CCDs
result = ccd_stack.merge_catalogs(good_indices='calibrate')
print('Total sources in merged catalog:', len(ccd_stack.catalog.sources))

psf = ccd_stack.create_psf(good_indices='psf', create_catalog=True,
    method='median')
ccd_stack.perform_psf_photometry()

ccd_stack.catalog.sources.write(
    '/media/data-beta/users/fmooleka/temp.ccd_stack.csv', format='ascii.csv')