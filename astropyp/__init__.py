# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    import astropyp.db_utils
    import astropyp.calibrate
    import astropyp.wrappers
    import astropyp.instruments
    import astropyp.utils
    import astropyp.catalog
    import astropyp.phot
    import astropyp.astrometry