from tb_calculations.extract_observables import (
    optical_cond_from_v,
    optical_cond_slow,
    optical_cond_tb_velocity,

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
def read_experimental(file_n, file_k, d_angst):
    data = np.loadtxt(fname=file_k, skiprows=1)
    lambda_mum = data[:,0]
    k = np.flip(data[:,1])
    data_n = np.loadtxt(fname=file_n, skiprows=1)
    n = np.flip(data_n[:,1])
    omega_SI = np.flip(2 * np.pi * sc.constants.c/(lambda_mum * 1e-6)) # omega = 2*  pi * f = 2 * pi * c/lambda
    omega_eV = omega_SI * sc.constants.physical_constants['hertz-electron volt relationship'][0] /(2*np.pi)
    eps = (n + 1j*k)**2
    re_sigma_2d_SI = np.real(-1j * omega_SI * sc.constants.epsilon_0 * d_angst*1e-10 * (eps - 1))
    A = re_sigma_2d_SI /(sc.constants.epsilon_0 * sc.constants.c)
    return omega_eV, re_sigma_2d_SI, A

omega_ref1, sigma_ref1, A_ref1 = read_experimental(file_n='data/experimental_spectra/islam_mos2_n.txt', file_k='data/experimental_spectra/islam_mos2_k.txt', d_angst=6.1)
omega_ref2, sigma_ref2, A_ref2 = read_experimental(file_n='data/experimental_spectra/ermolaev_mos2_n.txt', file_k='data/experimental_spectra/ermolaev_mos2_k.txt', d_angst=7)
omega_ref3, sigma_ref3, A_ref3 = read_experimental(file_n='data/experimental_spectra/ullberg_mos2_n.txt', file_k='data/experimental_spectra/ullberg_mos2_k.txt', d_angst=6.5)
omega_ref4, sigma_ref4, A_ref4 = read_experimental(file_n='data/experimental_spectra/zhang_mos2_n.txt', file_k='data/experimental_spectra/zhang_mos2_k.txt', d_angst=6.3)


    

lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb("./seedname_tb.dat") 
kFrac = k_grid(n_points=(120,120,1))
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
lattice_w, cells_w, degeneracies_w, Hr_w, Sr_w, vr_w = parse_and_FT.read_tb_momentum(fname='./data/martin_qe_mos2_ref/MoS2_custom_tb.dat', orthogonal=True)
omega_wannier, sigma_wannier = optical_cond_tb_velocity(Sr=Sr_w, Hr=Hr_w, vr=vr_w, kFrac=kFrac, cells=cells_w, valence_num=7, lattice_au=lattice_w, degeneracies=degeneracies_w, eta_eV=0.1, range_omega=(1,10), T_K=300)
# plot 3d conductivity in SI
omega_eV = au_to_eV(omega)
re_sigma_3d = np.real(sigma)
re_sigma_3d_SI = cond_3D_au_to_SI(re_sigma_3d)
omega_w_eV = au_to_eV(omega_wannier)
re_sigma_w_3d = np.real(sigma_wannier)
re_sigma_w_3d_SI = cond_3D_au_to_SI(re_sigma_w_3d)

fig, ax = plt.subplots(figsize=(6,4.5), facecolor='#F2F2F2')
ax.plot(omega_eV, re_sigma_3d_SI[:,0,0], label=r'$\sigma_{xx}, \sigma_{yy}$')
ax.plot(omega_eV, re_sigma_3d_SI[:,0,1], label=r'$\sigma_{xy}, \sigma_{yx}$')
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
ax.plot(omega_eV, re_sigma_2d[:,0,0], label=r'$\sigma_{xx}, \sigma_{yy}$')
ax.plot(omega_eV, re_sigma_2d[:,0,1], label=r'$\sigma_{xy}, \sigma_{yx}$')
#reference from quantum espresso
re_sigma_w_2d = np.real(lattice_w[2,2] * sigma_wannier)
scale = np.max(re_sigma_2d[:,0,0]) / np.max(re_sigma_w_2d[:,0,0])
print(scale)
ax.plot(omega_w_eV, re_sigma_w_2d[:,0,0], '--', color='blue', label=r'$\sigma_{xx}$ Quantum Espresso')
ax.plot(omega_w_eV, re_sigma_w_2d[:,0,1], '--', color='orange', label=r'$\sigma_{xy}$ Quantum Espresso')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\text{Re}(\sigma^{2D}) \ [\text{a.u.}]$")
ax.set_facecolor('#F2F2F2')
ax.legend(loc='upper right',facecolor='#F2F2F2')
plt.savefig("sigma_2d_au.pdf")
plt.show()

#plot sheet conductivity in SI
re_sigma_2d_SI = cond_2D_au_to_SI(re_sigma_2d)
re_sigma_w_2d_SI = cond_2D_au_to_SI(re_sigma_w_2d)
fig, ax = plt.subplots(figsize=(6,4.5),facecolor='#F2F2F2')
ax.plot(omega_eV, re_sigma_2d_SI[:,0,0], label=r'$\sigma_{xx}, \sigma_{yy}$')
ax.plot(omega_eV, re_sigma_2d_SI[:,0,1], label=r'$\sigma_{xy}, \sigma_{yx}$')
# ax.plot(omega_w_eV, re_sigma_w_2d_SI[:,0,0], '--', color='blue', label=r'$\sigma_{xx}$ Quantum Espresso')
# ax.plot(omega_w_eV, re_sigma_w_2d_SI[:,0,1], '--', color='orange', label=r'$\sigma_{xy}$ Quantum Espresso')
ax.plot(omega_ref1, sigma_ref1)
ax.plot(omega_ref2, sigma_ref2)
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"$\text{Re}(\sigma^{2D}) \ [S]$")
ax.legend(loc='upper right',facecolor='#F2F2F2')
ax.set_facecolor('#F2F2F2')
plt.savefig("sigma_2d_SI.pdf")
plt.show()

#plot sheet 
omega_si = omega_au_SI(omega_eV)
epsilon_SI = sigma_to_eps(sigma_si=cond_3D_au_to_SI(sigma), omega_si=omega_si)
absorbance_2d_SI = bohr_to_angstrom(lattice[2,2]) * 1e-10 * re_sigma_3d_SI/(sc.constants.epsilon_0 * sc.constants.c)
omega_w_SI = omega_au_SI(omega_w_eV)
epsilon_w_SI = sigma_to_eps(sigma_si=cond_3D_au_to_SI(sigma_wannier), omega_si=omega_w_SI)
absorbance_w_2d_SI = bohr_to_angstrom(lattice_w[2,2]) * 1e-10 * re_sigma_w_3d_SI/(sc.constants.epsilon_0 * sc.constants.c)
fig, ax = plt.subplots(figsize=(6,4.5), facecolor='#F2F2F2')
ax.plot(omega_eV, absorbance_2d_SI[:,0,0], label=r'DFTB')
# ax.plot(omega_eV, absorbance_2d_SI[:,0,1], label=r'$\sigma_{xy}, \sigma_{yx}$')
ax.plot(omega_w_eV, absorbance_w_2d_SI[:,0,0], label=r'Quantum ESPRESSO')
# ax.plot(omega_w_eV, absorbance_w_2d_SI[:,0,1], '--', color='orange', label=r'$\sigma_{xy}$ Quantum Espresso')
# ax.plot(omega_ref1, A_ref1, label='Experiment: Islam')
# ax.plot(omega_ref2, A_ref2, label='Experiment: Ermolaev')
# ax.plot(omega_ref3, A_ref3, label='Experiment: Ullberg')
# ax.plot(omega_ref4, A_ref4, label='Zhang')
ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
ax.set_ylabel(r"Absorbance")
ax.legend(loc='upper right',facecolor='#F2F2F2')
ax.set_facecolor('#F2F2F2')
plt.savefig("absorbance_2d.pdf")
plt.show()