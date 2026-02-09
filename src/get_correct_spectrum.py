from from_tb.extract_properties_tbfile import gradient_H_atomic, get_momentum_bloch, to_bloch_basis, get_momentum_orth, get_momentum_bloch_lee, absorption_spec, absorption_spec_lee
from from_tb.utils import k_grid, k_grid_bz
from from_tb.extract_properties_tbfile import gradient_H_atomic
import from_tb.w90 as w90
import numpy as np
import matplotlib.pyplot as plt

sigma_old = np.load(file='sigma_tens.npy')
omega = np.linspace(start=0, stop=1, num=np.shape(sigma_old)[0])
plt.plot(omega, sigma_old[:,0,0])
plt.plot(omega, sigma_old[:,0,1])
plt.plot(omega, sigma_old[:,1,0])
plt.plot(omega, sigma_old[:,1,1])
# plt.show()

lattice, cells, degeneracies, Hr, Sr, Rr = w90.read_tb("seedname_mos2_ase.dat") 
kFrac = k_grid(n_points=(10,10,1))
# kFrac = k_grid_bz(lattice=lattice, shape=(10,10,1))
# gradH_analytic = w90.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kFrac, lattice=lattice)
# gradH_numeric = gradient_H_atomic(Hr=Hr, kPoints=kFrac, cells=cells, lattice=lattice, degeneracies=degeneracies)
# 
# print(np.allclose(gradH_analytic, gradH_numeric))
# 
# """try Paredes formula"""
# momentum_bloch = get_momentum_bloch(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
# 
# """go via orthogonal basis"""
# pk_orth, Sk_orth, Hk_orth = get_momentum_orth(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
# momentum_from_orth, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk_orth, Hk_orth=Hk_orth, Sk_orth=Sk_orth) 
# 
# print(f"Paredes and own formula give same result: {np.allclose(momentum_bloch, momentum_from_orth)}")
# 
# """try Lee formula"""
# momentum_bloch_lee = get_momentum_bloch_lee(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
# 
# print(f"Paredes and Lee give same result: {np.allclose(momentum_bloch, momentum_bloch_lee)}")
# 
# omega, sigma_tens = absorption_spec(Sr=Sr, Hr=Hr, Rr=Rr, kPoints=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies,
#                                     range_omega=(0, 10),
#                                     valence_idx=8,
#                                     gamma_eV=0.1, 
#                                     eta_eV=0.1, 
#                                     )
# print(np.shape(sigma_tens))
# print(sigma_tens[2000])
absorption_spec_lee(Sr=Sr, Hr=Hr, Rr=Rr, kFrac=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies, valence_num=9, eta_eV=0.1)
