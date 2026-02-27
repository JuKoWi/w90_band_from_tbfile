from from_tb.extract_properties_tbfile import gradient_H_atomic, get_momentum_bloch, to_bloch_basis, get_momentum_orth, get_momentum_bloch_lee, abs_spec_inter_intra_separate, absorption_spec_lee
from from_tb.utils import k_grid, k_grid_bz
from from_tb.extract_properties_tbfile import gradient_H_atomic
import from_tb.parse_and_FT as parse_and_FT
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'


lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb("seedname_tb_transpose.dat") 
kFrac = k_grid(n_points=(10,10,1))
# kFrac = k_grid_bz(lattice=lattice, shape=(40,40,1))
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
omega, sigma = absorption_spec_lee(Sr=Sr, Hr=Hr, Rr=Rr, kFrac=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies, valence_num=9, eta_eV=0.1)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(omega, sigma[:,0,0], label=r'$\sigma_{x,x}$')
ax.plot(omega, sigma[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
# ax.plot(omega, sigma[:,1,0], label=r'$\sigma_{y,x}$')
ax.plot(omega, sigma[:,1,1], label=r'$\sigma_{y,y}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\sigma / max(\sigma)$")
ax.legend(loc='upper right')
plt.savefig("plot_sigma.pdf")
plt.show()
print(sigma[2000])


