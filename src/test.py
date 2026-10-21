from tb_calculations.parse_and_FT import parse_hdf5, read_tb, read_tb_momentum
from abs_spec_momentum import compare_spectra
import numpy as np

spectra = ['onsite_expansion-cond.hdf5', 'sk-cond.hdf5', 'mulliken-cond.hdf5', 'shift_only-cond.hdf5', 'onsite_only-cond.hdf5']
compare_spectra(specfiles=spectra)
