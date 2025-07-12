__version__ = '0.0.4'
from .xabba import ABBA, XABBA, fastXABBA, fastXABBA_len, fastXABBA_inc
import warnings

try:
    # # %load_ext Cython
    # !python3 setup.py build_ext --inplace
    from .compfp import compress
    from .aggfp import aggregate 
    from .inversefp import *
        
except ModuleNotFoundError:
    warnings.warn("cython fail.")
    from .comp import compress
    from .agg import aggregate
    from .inverse import *

from .llmabba import LLMABBA

__name__ =  'llmabba'
