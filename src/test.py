from tb_calculations.parse_and_FT import parse_hdf5, read_tb, read_tb_momentum, read_dftb_hamiltonian, hdf5_hamreal
import numpy as np

lat_dict = read_dftb_hamiltonian(file='hamreal1.dat')
hdf5_hamreal(hamreal='hamreal1.dat', hdf5file='ZnO_zincblende_sk.hdf5', outfile='modified_zincblende.hdf5')
# hdf5_hamreal(hamreal='hamreal1.dat', hdf5file='mos2_sk.hdf5', outfile='modified_mos2.hdf5')

# tb1 = parse_hdf5(filename='modified_zincblende.hdf5')
# tb2 = parse_hdf5(filename='ZnO_zincblende_sk.hdf5')
# for i in range(tb1['H'].shape[0]):
#     print(np.allclose(tb1['H'][i], tb2['H'][i], rtol=1e-4))
#     if np.allclose(tb1['lat_points'][i], [0,0,0]):
#         zerocell = i
# print(tb1['lat_points'])
# print(np.real(tb1['H'])[zerocell][4:10,4:10])
# print(tb2['H'][zerocell][4:10,4:10])
# print(read_dftb_hamiltonian(file='hamreal1.dat')[(0,0,0)][4:10,4:10])



# spectra = ['onsite_expansion-cond.hdf5', 'sk-cond.hdf5', 'mulliken-cond.hdf5', 'shift_only-cond.hdf5', 'onsite_only-cond.hdf5']
# compare_spectra(specfiles=spectra)
