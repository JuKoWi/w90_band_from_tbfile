from tb_calculations.extract_observables import (
    optical_cond_from_v,
    optical_cond_slow,

)
from tb_calculations.utils import (
    k_grid, 
    cond_2D_au_to_SI,
    cond_3D_au_to_SI,
    sigma_to_eps,
    au_to_eV,
    omega_au_SI,
    angstrom_to_bohr,
    bohr_to_angstrom,
)
import tb_calculations.parse_and_FT as parse_and_FT
import numpy as np
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import scipy as sc
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

#experimental reference
def read_experimental(file):
    data = np.loadtxt(fname=file, skiprows=1)
    lambda_mum = data[:,0]
    k = data[:,1]
    omega = sc.constants.c/(lambda_mum * 1e-6) # omega = 2*  pi * f = 2 * pi * c/lambda
    omega_eV = omega * sc.constants.physical_constants['hertz-electron volt relationship'][0]
    return np.flip(omega_eV), np.flip(k)

# omega_ref1, sigma_ref1 = read_experimental(file='data/experimental_spectra/islam_mos2_k.txt')


    

lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb("./seedname_tb.dat") 
kFrac = k_grid(n_points=(100,100,1))
path_sigma = Path("sigma.npy")
path_omega = Path("omega.npy")
if path_omega.exists() and path_sigma.exists():
    omega = np.load(file=path_omega)
    sigma = np.load(file=path_sigma)
else:
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

    np.save(file="omega", arr=omega)
    np.save(file='sigma', arr=sigma)
# plot 3d conductivity in SI
omega_eV = au_to_eV(omega)
re_sigma_3d = np.real(sigma)
re_sigma_3d_SI = cond_3D_au_to_SI(re_sigma_3d)
fig, ax = plt.subplots(figsize=(6,4.5), facecolor='#F2F2F2')
ax.plot(omega_eV, re_sigma_3d_SI[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y}$')
ax.plot(omega_eV, re_sigma_3d_SI[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\text{Re}(\sigma) \ [S/m]$")
ax.legend(loc='upper right', facecolor='#F2F2F2')
ax.set_facecolor('#F2F2F2')
plt.savefig("sigma_3d_SI.pdf")
plt.show()

#plot sheet conductivity in a.u.
sigma_2d = lattice[2,2] * sigma
re_sigma_2d = np.real(sigma_2d)
fig, ax = plt.subplots(figsize=(6,4.5),facecolor='#F2F2F2')
ax.plot(omega_eV, re_sigma_2d[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y}$')
ax.plot(omega_eV, re_sigma_2d[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\text{Re}(\sigma^{2D}) \ [\text{a.u.}]$")
ax.set_facecolor('#F2F2F2')
ax.legend(loc='upper right',facecolor='#F2F2F2')
plt.savefig("sigma_2d_au.pdf")
plt.show()

#plot sheet conductivity in SI
re_sigma_2d_SI = cond_2D_au_to_SI(re_sigma_2d)
fig, ax = plt.subplots(figsize=(6,4.5),facecolor='#F2F2F2')
ax.plot(omega_eV, re_sigma_2d_SI[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y}$')
ax.plot(omega_eV, re_sigma_2d_SI[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\text{Re}(\sigma^{2D}) \ [S]$")
ax.legend(loc='upper right',facecolor='#F2F2F2')
ax.set_facecolor('#F2F2F2')
plt.savefig("sigma_2d_SI.pdf")
plt.show()

#plot sheet 
omega_si = omega_au_SI(omega_eV)
epsilon_SI = sigma_to_eps(sigma_si=cond_3D_au_to_SI(sigma), omega_si=omega_si)
# absorbance_2d_SI = omega_si * bohr_to_angstrom(lattice[2,2]) * 1e-10 * np.imag(epsilon_SI) / sc.constants.c
absorbance_2d_SI = bohr_to_angstrom(lattice[2,2]) * 1e-10 * re_sigma_3d_SI/(sc.constants.epsilon_0 * sc.constants.c)
fig, ax = plt.subplots(figsize=(6,4.5), facecolor='#F2F2F2')
ax.plot(omega_eV, absorbance_2d_SI[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y}$')
ax.plot(omega_eV, absorbance_2d_SI[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$A(\omega)$")
ax.legend(loc='upper right',facecolor='#F2F2F2')
ax.set_facecolor('#F2F2F2')
plt.savefig("absorbance_2d.pdf")
plt.show()