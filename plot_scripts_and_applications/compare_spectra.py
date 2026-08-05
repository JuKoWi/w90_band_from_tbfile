"""generate absorption spectrum plots comparing different ways/methods to generate the absorption spectrum"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tb_calculations.extract_observables import gradient_H_atomic, get_momentum_bloch, to_bloch_basis, get_momentum_orth, get_momentum_bloch_lee, abs_spec_inter_intra_separate, absorption_spec_lee, absorption_spec_momentum
from tb_calculations.utils import k_grid, k_grid_bz
from tb_calculations.extract_observables import gradient_H_atomic
import tb_calculations.parse_and_FT as parse_and_FT
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

"""for cubic and rectangle spectrum correct if atom on origin, not correct if not"""
# seedname_dir = './seedname_pseudocarb/'
# carbon_structure = 'cube/'
# file1 = seedname_dir + carbon_structure + 'seedname_tb.dat'
# file2 = seedname_dir + carbon_structure + 'seedname_tb_momentum.dat'
# carbon_structure = 'cube_origin'
# file2 = seedname_dir + carbon_structure + 'seedname_tb.dat'

"""origin shift or no origin shift makes no difference"""
seedname_dir = './seedname_mos2/'
file1 = seedname_dir + 'seedname_tb.dat'
# file2 = seedname_dir + 'seedname_tb_momentum.dat'


# seedname_dir = './seedname_graphene/'
# file1 = seedname_dir +  'seedname_tb.dat'
# file2 = seedname_dir +  'seedname_tb_momentum.dat'

def compare_momentum_velocity(file_velocity, file_momentum, valence_num, ktuple):
    kFrac = k_grid(n_points=ktuple)
    lattice1, cells1, degeneracies1, Hr1, Sr1, Rr1 = parse_and_FT.read_tb(file_velocity) 
    omega1, sigma1 = absorption_spec_lee(Sr=Sr1, Hr=Hr1, Rr=Rr1, kFrac=kFrac, lattice=lattice1, cells=cells1, degeneracies=degeneracies1, valence_num=valence_num, eta_eV=0.1, range_omega=(0,10))

    lattice2, cells2, degeneracies2, Hr2, Sr2, pr2 = parse_and_FT.read_tb_momentum(file_momentum) 
    omega2, sigma2 = absorption_spec_momentum(Sr=Sr2, Hr=Hr2, pr=pr2, kFrac=kFrac, lattice=lattice2, cells=cells2, degeneracies=degeneracies2, valence_num=valence_num, eta_eV=0.1)
        
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.plot(omega1, sigma1[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y} \ \mathbf{v}$')
    ax.plot(omega2, sigma2[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y} \ \mathbf{p}$')

    ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
    ax.set_ylabel(r"$\Re{(\sigma)}/ max[\Re{(\sigma)}]$")
    ax.legend(loc='upper right')

    plt.savefig("plot_sigma.pdf")
    plt.show()

def compare_velocity(file1, file2, valence_num, ktuple):
    kFrac = k_grid(n_points=ktuple)
    lattice1, cells1, degeneracies1, Hr1, Sr1, Rr1 = parse_and_FT.read_tb(file1) 
    omega1, sigma1 = absorption_spec_lee(Sr=Sr1, Hr=Hr1, Rr=Rr1, 
                                         kFrac=kFrac, 
                                         lattice=lattice1, 
                                         cells=cells1, 
                                         degeneracies=degeneracies1, 
                                         valence_num=valence_num, 
                                         eta_eV=0.1, 
                                         range_omega=(0,20), 
                                         normalized=False
                                         )

    lattice2, cells2, degeneracies2, Hr2, Sr2, Rr2 = parse_and_FT.read_tb(file2, orthogonal=True) 
    omega2, sigma2 = absorption_spec_lee(Sr=Sr2, Hr=Hr2, Rr=Rr2, 
                                         kFrac=kFrac, 
                                         lattice=lattice2,
                                         cells=cells2, 
                                         degeneracies=degeneracies2, 
                                         valence_num=valence_num, 
                                         eta_eV=0.1, 
                                         range_omega=(0,20),
                                         normalized=False
                                         )
        
    fig, axs = plt.subplots(ncols=2, figsize=(14.5,4.5))
    axs[0].plot(omega1, sigma1[:,0,0], label=r'$\sigma_{x,x}$')
    axs[0].plot(omega2, sigma2[:,0,0], label=r'$\sigma_{x,x}$')
    axs[0].plot(omega1, sigma1[:,1,1], label=r'$\sigma_{y,y}$')
    axs[0].plot(omega2, sigma2[:,1,1], label=r'$\sigma_{y,y}$')

    axs[1].plot(omega1, sigma1[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
    axs[1].plot(omega2, sigma2[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
    axs[0].set_xlabel(r"$\omega \ [\mathrm{eV}]$")
    axs[0].set_ylabel(r"$\sigma / max(\sigma)$")
    axs[0].legend(loc='upper right')
    axs[1].set_xlabel(r"$\omega \ [\mathrm{eV}]$")
    axs[1].set_ylabel(r"$\sigma / max(\sigma)$")
    axs[1].legend(loc='upper right')

    plt.savefig("plot_sigma.pdf")
    plt.show()

def plot_single(velocity:bool, file, valence_num, ktuple):
    """Plot the polarizability tensor components for a single method"""
    kFrac = k_grid(n_points=ktuple)
    if velocity:
        lattice1, cells1, degeneracies1, Hr1, Sr1, Rr1 = parse_and_FT.read_tb(file) 
        omega1, sigma1 = absorption_spec_lee(Sr=Sr1, Hr=Hr1, Rr=Rr1, kFrac=kFrac, lattice=lattice1, cells=cells1, degeneracies=degeneracies1, valence_num=valence_num, eta_eV=0.1, range_omega=(0,10), T_K=0)
    else:
        lattice1, cells1, degeneracies1, Hr1, Sr1, pr1 = parse_and_FT.read_tb_momentum(file) 
        omega1, sigma1 = absorption_spec_momentum(Sr=Sr1, Hr=Hr1, pr=pr1, kFrac=kFrac, lattice=lattice1, cells=cells1, degeneracies=degeneracies1, valence_num=valence_num, eta_eV=0.1)

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.plot(omega1, sigma1[:,0,0], label=r'$\sigma_{x,x}, \sigma_{y,y}$')
    ax.plot(omega1, sigma1[:,0,1], label=r'$\sigma_{x,y}, \sigma_{y,x}$')
    # ax.plot(omega1, sigma1[:,1,1], label=r'$\sigma_{y,y}$')
    # ax.plot(omega1, sigma1[:,1,0]) 
    ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
    ax.set_ylabel(r"$\Re{(\sigma)}/ max[\Re{(\sigma)}]$")
    ax.legend(loc='upper right')
    plt.savefig("plot_sigma.pdf")
    plt.show()

# compare_momentum_velocity(file_velocity=file1, file_momentum=file2, valence_num=4, ktuple=(220,220,1))
# compare_velocity(file1=file1, file2='seedname_orth.dat', valence_num=9, ktuple=(10,10,1))
plot_single(velocity=True, file='seedname_tb.dat', valence_num=2, ktuple=(30, 30,1))


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


