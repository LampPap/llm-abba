__version__ = '0.0.5'
print("check1")
from .xabba import ABBA, XABBA, fastXABBA, fastXABBA_len, fastXABBA_inc
import warnings

print("check2")
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
