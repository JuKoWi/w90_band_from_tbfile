from from_tb.parse_and_FT import read_tb_momentum, read_tb
from from_tb.extract_properties_tbfile import absorption_spec_file
from from_tb.utils import k_grid
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

lattice, cells, degeneracies, Hr, Sr, pr = read_tb_momentum("seedname_tb_momentum.dat") 
kFrac = k_grid(n_points=(10,10,1))
omega, sigma = absorption_spec_file(Sr=Sr, Hr=Hr, pr=pr, kFrac=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies, valence_num=9, eta_eV=0.1)
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