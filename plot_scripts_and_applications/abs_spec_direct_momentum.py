"""Generate absorption spectrum plot based on momentum matrix elements"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tb_calculations.parse_and_FT import read_tb_momentum, read_tb
from tb_calculations.extract_observables import absorption_spec_momentum
from tb_calculations.utils import k_grid
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'
lattice, cells, degeneracies, Hr, Sr, pr = read_tb_momentum("seedname_mos2/seedname_tb_momentum.dat") 
kFrac = k_grid(n_points=(40,40,1))
omega, sigma = absorption_spec_momentum(Sr=Sr, Hr=Hr, pr=pr, kFrac=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies, valence_num=9, eta_eV=0.1)
fig, ax = plt.subplots(figsize=(6,4.5))
ax.plot(omega, sigma[:,0,0], label=r'$\sigma_{x,x}$')
ax.plot(omega, sigma[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
# ax.plot(omega, sigma[:,1,0], label=r'$\sigma_{y,x}$')
ax.plot(omega, sigma[:,1,1], label=r'$\sigma_{y,y}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\Re{\left(\sigma\right)} / max(\Re{\left(\sigma\right)})$")
ax.legend(loc='upper right')
plt.savefig("plot_sigma.pdf")
plt.show()
print(sigma[2000])