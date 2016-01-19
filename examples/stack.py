from __future__ import division, print_function
import time
from collections import OrderedDict
times = OrderedDict()
times['0'] = time.time()
import numpy as np
import logging

from astropy.io import fits
import astropy.wcs

# Astropy gives a lot of obnxious warnings that
# are difficult to filter individually
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

import astropyp
from astropyp import phot

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

times['init'] = time.time()

obj='F100'
min_flux = 1000
min_amplitude = 1000
good_amplitude = 50
calibrate_amplitude = 200
frame = 1
#explist = [442430, 442431, 442432]
explist = [442433, 442434, 442435]
aper_radius = 8
gain=4.
exptime=30.
max_offset=3

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
times['load all fits'] = time.time()

# Detect sources
ccd_stack = phot.stack.Stack(ccds, ref_index=1)
ccd_stack.detect_sources(min_flux=min_flux, good_amplitude=good_amplitude,
    calibrate_amplitude=calibrate_amplitude, psf_amplitude=1000, 
    sex_params=sex_params, subtract_bkg=True, windowed=False)

times['detect'] = time.time()
# Get astrometric solution
ccd_stack.get_transforms()
times['transforms'] = time.time()

#Stack images
stack, dqmask = ccd_stack.stack(slices=[slice(0,500), slice(0,500)])
times['stack'] = time.time()

#Save Coadd and Dqmask
primary_hdu = fits.PrimaryHDU()
img_hdu = fits.CompImageHDU(stack)
hdulist = fits.HDUList([primary_hdu, img_hdu])
hdulist.writeto('/media/data-beta/users/fmooleka/temp/test_stack.fits',
    clobber=True)

primary_hdu = fits.PrimaryHDU()
img_hdu = fits.CompImageHDU(dqmask)
hdulist = fits.HDUList([primary_hdu, img_hdu])
hdulist.writeto('/media/data-beta/users/fmooleka/temp/test_stack.dqmask.fits',
    clobber=True)
times['save'] = time.time()

# Detect sources in the stack
ccd_stack.detect_sources(ccd_stack.stack, min_flux=min_flux, 
    good_amplitude=good_amplitude,
    calibrate_amplitude=calibrate_amplitude, psf_amplitude=1000, 
    sex_params=sex_params, subtract_bkg=True, windowed=False)
times['stack detect'] = time.time()

# Create the PSF for the stacked image
ccd_stack.stack.create_psf()
times['create PSF'] = time.time()

#ccd.show_psf()
#good_idx = ccd.catalog.sources['peak']>calibrate_amplitude
good_idx = ccd_stack.stack.catalog.sources['peak']>good_amplitude
good_idx = good_idx & (ccd_stack.stack.catalog.sources['pipeline_flags']==0)
result = ccd_stack.stack.perform_psf_photometry(indices=good_idx)
times['PSF photometry'] = time.time()

time_keys = times.keys()
for n in range(1,len(time_keys)):
    key = time_keys[n]
    print('{0}: {1:.2f}s'.format(key, times[key]-times[time_keys[n-1]]))
print('Total time: {0:.2f}'.format(times[time_keys[-1]]-times[time_keys[0]]))

ccd = ccd_stack.stack

#good_idx = ccd.catalog.sources['peak']>calibrate_amplitude
good_idx = ccd.catalog.sources['peak']>good_amplitude
good_idx = good_idx & (ccd.catalog.sources['pipeline_flags']==0)
good_idx = good_idx & np.isfinite(ccd.catalog.sources['psf_mag'])
good_sources = ccd.catalog.sources[good_idx]

print('good sources')
print('rms', np.sqrt(np.sum(good_sources['psf_mag_err']**2/len(good_sources))))
print('mean', np.mean(good_sources['psf_mag_err']))
print('median', np.median(good_sources['psf_mag_err']))
print('stddev', np.std(good_sources['psf_mag_err']))

bad_count = np.sum(good_sources['psf_mag_err']>.05)
print('bad psf error: {0}, or {1}%'.format(
    bad_count, bad_count/len(good_sources)*100))
print('Better than 5%: {0} of {1}'.format(
    np.sum(good_sources['psf_mag_err']<=.05), len(good_sources)))
print('Better than 2%: {0} of {1}'.format(
    np.sum(good_sources['psf_mag_err']<=.02), len(good_sources)))
good_sources['aper_flux','psf_flux','peak','psf_mag_err'][
    good_sources['psf_mag_err']>.05]

good_idx = ccd.catalog.sources['peak']>calibrate_amplitude
good_idx = good_idx & (ccd.catalog.sources['pipeline_flags']==0)
good_idx = good_idx & np.isfinite(ccd.catalog.sources['psf_mag'])
good_sources = ccd.catalog.sources[good_idx]

print('Calibrate Sources')
print('rms', np.sqrt(np.sum(good_sources['psf_mag_err']**2/len(good_sources))))
print('mean', np.mean(good_sources['psf_mag_err']))
print('median', np.median(good_sources['psf_mag_err']))
print('stddev', np.std(good_sources['psf_mag_err']))

bad_count = np.sum(good_sources['psf_mag_err']>.05)
print('bad psf error: {0}, or {1}%'.format(
    bad_count, bad_count/len(good_sources)*100))
print('Better than 5%: {0} of {1}'.format(
    np.sum(good_sources['psf_mag_err']<=.05), len(good_sources)))
print('Better than 2%: {0} of {1}'.format(
    np.sum(good_sources['psf_mag_err']<=.02), len(good_sources)))
good_sources['aper_flux','psf_flux','peak','psf_mag_err'][
    good_sources['psf_mag_err']>.05]