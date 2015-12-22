from collections import OrderedDict
import numpy as np

from astropy.extern import six
from astropy import table

def isnumber(x):
    try:
        float(x)
    except (ValueError, TypeError) as e:
        return False
    return True

uint8_flags = OrderedDict([
    (128, 'Bit 8'),
    (64, 'Bit 7'),
    (32, 'Bit 6'),
    (16, 'Bit 5'),
    (8, 'Bit 4'),
    (4, 'Bit 3'),
    (2, 'Bit 2'),
    (1, 'Bit 1')
])

class InternalFlags:
    """
    InternalFlags must be initialized with an OrderedDict, such
    as uint8_flags.  
    """
    def __init__(self, flags=uint8_flags):
        self.flags = flags
        if isinstance(flags, six.string_types):
            self.set_flags(flags)
        self.flag_type = self.set_flag_type()
    def set_flag_type(self):
        bit_keys = [1,2,4,8,16,32,64,128]
        if np.all([k in bit_keys for k in self.flags.keys()]):
            self.flag_type = 'bits'
        else:
            self.flag_type = 'values'
        return self.flag_type
    def set_flags(self, flag_module):
        if flag_module=='sex':
            import astropyp.wrappers.astromatic.utils as utils
            self.flags = utils.sex_internal_flags
        elif flag_module=='decam early':
            import astropyp.instruments.decam.utils as utils
            self.flags = utils.decam_flags_early
        elif flag_module=='decam':
            import astropyp.instruments.decam.utils as utils
            self.flags = utils.decam_flags
        else:
            raise Exception("Flag {0} has not yet been added".format())
        self.set_flag_type()
    def get_flag_info(self):
        if self.flag_type=='bits':
            key = 'Bits'
        else:
            key = 'Value'
        result = table.Table(
            [self.flags.keys(), [v for k,v in self.flags.items()]], 
            names=(key, 'Description'))
        return result
    def get_flags(self, flags):
        if self.flag_type=='bits':
            binflags = flags.astype(np.uint8)
            binflags = binflags.reshape((binflags.shape[0],1))
            binflags = np.unpackbits(binflags, axis=1)
            tbl = table.Table()
            for n,f in enumerate(self.flags):
                tbl[str(f)] = np.array(binflags[:,n], dtype=bool)
        else:
            tbl = table.Table()
            for n,f in enumerate(self.flags):
                tbl[str(f)] = flags==f
        return tbl

def update_ma_idx(arr, idx):
    new_array = np.ma.array(arr)[idx]
    new_array.mask = new_array.mask | idx.mask
    return new_array