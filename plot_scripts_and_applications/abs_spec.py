from tb_calculations.extract_observables import abs_spec_inter_intra_separate
from tb_calculations.utils import k_grid_bz, k_grid
import tb_calculations.parse_and_FT as parse_and_FT
import numpy as np
import matplotlib.pyplot as plt

# segments, labels, bands_orth, H_orth, S_orth, d_orth =  bandstructure_orth_basis(lattice, cells, Hr, Sr, Rr)
# bands_alex = parse_dftb_band(filepath="band_mos2_alex_27band.out", n_bands=27) 
# bands_alex = [bands_alex[:100], bands_alex[100:200], bands_alex[200:300]]
# bands_own = parse_dftb_band(filepath="band_mos2_own_27band_denssup_corrected_eigval.out", n_bands=27)
# bands_own = [bands_own[:100], bands_own[100:200], bands_own[200:300]]

# lattice, cells, degeneracies2, Hr, Sr, Rr = w90.read_tb("seedname_mos2_ase.dat")
lattice, cells, degeneracies2, Hr, Sr, Rr = parse_and_FT.read_tb("seedname_tb_pseudocarb.dat")
# dk1, Sk_orth, Hk_orth= get_dipole(degeneracies=degeneracies2, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=[[0.2, 0, 0]], cells=cells, lattice=lattice)
# dk2, Sk_orth, Hk_orth= get_dipole(degeneracies=degeneracies2, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=[[-0.2, 0, 0]], cells=cells, lattice=lattice)
# dkA = dk1[0,1,0]
# dkB = dk2[0,1,0]
# print(dkA)
# print(dkB)
# sys.exit()
kcart1 = np.array([0.5,0,0])
kcart2 = np.array([-0.5,0,0])
kfrac1 = 1/(2*np.pi) * lattice @ kcart1
kfrac2 = 1/(2*np.pi) * lattice @ kcart2
omega, sigma_tens2 = abs_spec_inter_intra_separate(Sr=Sr, 
                                    Hr=Hr, 
                                    Rr=Rr, 
                                    # kPoints=k_grid(n_points=(10, 10, 1)), 
                                    # kPoints=np.array([kfrac1, kfrac2 ]),
                                    kPoints=k_grid_bz(lattice=lattice, shape=(20,20,1)),
                                    lattice=lattice,
                                    cells=cells,
                                    range_omega=(0, 10),
                                    valence_idx=8,
                                    gamma_eV=0.1, 
                                    eta_eV=0.1, 
                                    degeneracies=degeneracies2)

# rot120deg = np.array([[-0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, -0.5]])
# sigma_tens2 = np.einsum('ab, obc, cd -> oad', rot120deg.T, sigma_tens2, rot120deg )
print(sigma_tens2[2000])
# print(sigma_tens2[2000])
# plt.plot(omega, spectrum_in_direction(sigma_tens=sigma_tens2, vec=[1,0]))
# plt.plot(omega, spectrum_in_direction(sigma_tens=sigma_tens2, vec=[-0.5,np.sqrt(3)/2]))
# plt.show()

# plt.plot(omega, sigma_tens[:,0,1], 
#          '.',
#            ms=1)
plt.plot(omega, sigma_tens2[:,1,0],
         '.',
           ms=1)
plt.plot(omega, sigma_tens2[:,0,1],
         '.',
           ms=1)
plt.show()


# plt.plot(omega, sigma_tens[:,1,0], 
#          '.',
#            ms=1)
plt.plot(omega, sigma_tens2[:,0,0],
         '.',
           ms=1)
plt.plot(omega, sigma_tens2[:,1,1],
         '.',
           ms=1)
plt.show()