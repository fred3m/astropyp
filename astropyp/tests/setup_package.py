import os

def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': [
            'coveragerc',
            os.path.join('data', 'fake_obs0.fits'),
            os.path.join('data', 'fake_obs1.fits'),
            os.path.join('data', 'fake_obs2.fits'),
            os.path.join('data', 'fake_obs3.fits'),
            os.path.join('data', 'fake_obs4.fits'),
            os.path.join('data', 'fake_obs5.fits'),
            os.path.join('data', 'fake_obs6.fits'),
            os.path.join('data', 'fake_obs7.fits'),
            os.path.join('data', 'fake_obs8.fits'),
            os.path.join('data', 'fake_obs9.fits')
        ],
    }