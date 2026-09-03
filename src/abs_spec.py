from tb_calculations.extract_observables import (
    optical_cond_from_v,

)
from tb_calculations.utils import (
    k_grid, 
    cond_au_to_SI,
    au_to_eV,
)
import tb_calculations.parse_and_FT as parse_and_FT
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'


lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb("seedname_tb.dat") 
kFrac = k_grid(n_points=(40,40,1))
omega, sigma = optical_cond_from_v(
                                Sr=Sr, 
                                Hr=Hr, 
                                Rr=Rr, 
                                kFrac=kFrac, 
                                lattice_au=lattice, 
                                cells=cells, 
                                degeneracies=degeneracies, 
                                valence_num=9, 
                                eta_eV=0.1,
                                T_K=300,
                                   )
sigma = cond_au_to_SI(np.real(sigma))
omega = au_to_eV(omega)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(omega, sigma[:,0,0], label=r'$\sigma_{x,x}$')
ax.plot(omega, sigma[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
# ax.plot(omega, sigma[:,1,0], label=r'$\sigma_{y,x}$')
ax.plot(omega, sigma[:,1,1], label=r'$\sigma_{y,y}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\sigma \ [S/m]$")
ax.legend(loc='upper right')
plt.savefig("plot_sigma.pdf")
plt.show()


