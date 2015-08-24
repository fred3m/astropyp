from __future__ import absolute_import
import os

def get_package_data():
    return {
        'astropyp.wrappers.astromatic.tests': [
            os.path.join('data', 'test.ldac.fits'),
            os.path.join('data', 'multiext.ldac.fits'),
        ]
    }