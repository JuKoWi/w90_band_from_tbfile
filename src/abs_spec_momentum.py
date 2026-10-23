"""Generate absorption spectrum plot based on momentum matrix elements"""
from tb_calculations.parse_and_FT import read_tb, read_tb_momentum, parse_hdf5, read_tb
from tb_calculations.extract_observables import optical_cond_from_p, optical_cond_from_r 
from tb_calculations.utils import k_grid, au_to_eV, cond_3D_au_to_SI, omega_au_SI, bohr_to_angstrom, sigma_to_eps
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import scipy as sc
import h5py
import sys

def generate_spectra(tb_files, ktuple, filled_bands, momentum=False):
    kFrac = k_grid(n_points=ktuple)
    eta_eV = 0.1
    for file in tb_files:
        tb_model = parse_hdf5(file)
        if momentum:
            omega, sigma = optical_cond_from_p(
                Sr=tb_model['S'],
                Hr=tb_model['H'],
                pr=tb_model['p'],
                kFrac=kFrac,
                cellvec_au=tb_model['cell'],
                lat_points=tb_model['lat_points'],
                degeneracies=tb_model['degen'],
                filled_bands=filled_bands,
                eta_eV=eta_eV,
                T_K=300,
            )
            with h5py.File(f'momentum-cond.hdf5', 'w') as f:
                f.attrs['approx'] = 'momentum'
                f.attrs['grid'] = ktuple
                f.attrs['eta_eV'] = eta_eV
                dset_omega = f.create_dataset('omega', data=omega)
                dset_omega.attrs['unit'] = 'a.u.'
                dset_sigma = f.create_dataset('sigma', data=sigma)
                dset_sigma.attrs['unit'] = 'a.u.'
        else:
            omega, sigma = optical_cond_from_r(
                                        Sr=tb_model['S'], 
                                        Hr=tb_model['H'], 
                                        Rr=tb_model['r'], 
                                        kFrac=kFrac, 
                                        cellvec_au=tb_model['cell'], 
                                        lat_points=tb_model['lat_points'], 
                                        degeneracies=tb_model['degen'], 
                                        filled_bands=filled_bands, 
                                        eta_eV=eta_eV,
                                        T_K=300,
            )
            with h5py.File(f'{tb_model['approx']}-cond.hdf5', 'w') as f:
                f.attrs['approx'] = tb_model['approx']
                f.attrs['grid'] = ktuple
                f.attrs['eta_eV'] = eta_eV
                dset_omega = f.create_dataset('omega', data=omega)
                dset_omega.attrs['unit'] = 'a.u.'
                dset_sigma = f.create_dataset('sigma', data=sigma)
                dset_sigma.attrs['unit'] = 'a.u.'

def compare_spectra(specfiles):
    fig, ax = plt.subplots()
    for spec in specfiles:
        with h5py.File(spec, 'r') as f:
            omega = np.array(f['omega'])
            sigma = np.array(f['sigma'])
            ax.plot(au_to_eV(omega), np.real(sigma[:,0,0]), label=f.attrs.get('approx'))
            print(np.allclose(np.real(sigma[:,0,0]), np.real(sigma[:,1,1])))
    ax.set_xlabel(r'$\omega$ [eV]')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    tb_files = ['graphene_mulliken.hdf5','graphene_onsite_expansion.hdf5', 'graphene_onsite_only.hdf5', 'graphene_shift_only.hdf5', 'graphene_sk.hdf5']
    single = ['graphene_sk.hdf5']
    generate_spectra(tb_files=single, ktuple=(30,30,1), filled_bands=4, momentum=True) 
    spectra = ['onsite_expansion-cond.hdf5', 'sk-cond.hdf5', 'mulliken-cond.hdf5', 'shift_only-cond.hdf5', 'onsite_only-cond.hdf5']
    lattice, cells, degeneracies, Hr, Sr, Rr = read_tb("./data/seedname_graphene/seedname_tb.dat") 

    kFrac = k_grid(n_points=(30,30,1))
    omega1, sigma1 = optical_cond_from_r(
                                    Sr=Sr, 
                                    Hr=Hr, 
                                    Rr=Rr, 
                                    kFrac=kFrac, 
                                    cellvec_au=lattice, 
                                    lat_points=cells, 
                                    degeneracies=degeneracies, 
                                    filled_bands=4, 
                                    eta_eV=0.1,
                                    T_K=300,
                                       )
    # compare_spectra(specfiles=spectra)
    fig, ax = plt.subplots()
    with h5py.File("momentum-cond.hdf5", 'r') as f:
        omega2 = np.array(f['omega'])
        sigma2 = np.array(f['sigma'])
    ax.plot(au_to_eV(omega1), np.real(sigma1[:,0,0]), label='old seedname momentum')
    print(np.allclose(np.real(sigma1[:,0,0]), np.real(sigma1[:,1,1])))
    ax.plot(au_to_eV(omega2), np.real(sigma2[:,0,0]), label='new hdf5')
    print(np.allclose(np.real(sigma2[:,0,0]), np.real(sigma2[:,1,1])))
    ax.set_xlabel(r'$\omega$ [eV]')
    ax.legend()
    plt.show()


    sys.exit()
    
    omega_eV = au_to_eV(omega1)
    re_sigma_3d = np.real(sigma1)
    re_sigma_3d_SI = cond_3D_au_to_SI(re_sigma_3d)
    omega_si = omega_au_SI(omega_eV)
    epsilon_SI = sigma_to_eps(sigma_si=cond_3D_au_to_SI(sigma1), omega_si=omega_si)
    absorbance_2d_SI = bohr_to_angstrom(lattice[2,2]) * 1e-10 * re_sigma_3d_SI/(sc.constants.epsilon_0 * sc.constants.c)
    
    
    omega_eV_p = au_to_eV(omega_p)
    re_sigma_3d_p = np.real(sigma_p)
    re_sigma_3d_SI_p = cond_3D_au_to_SI(re_sigma_3d_p) 
    omega_si_p = omega_au_SI(omega_eV_p)
    epsilon_SI_p = sigma_to_eps(sigma_si=cond_3D_au_to_SI(sigma_p), omega_si=omega_si_p)
    absorbance_2d_SI_p = bohr_to_angstrom(lattice[2,2]) * 1e-10 * re_sigma_3d_SI_p/(sc.constants.epsilon_0 * sc.constants.c)
    
    
    a2 = absorbance_2d_SI_p[:,0,0] * (np.max(absorbance_2d_SI[:,0,0])/np.max(absorbance_2d_SI_p[:,0,0]))
    fig, ax = plt.subplots(figsize=(16,9), facecolor='#F2F2F2')
    ax.plot(omega_eV, absorbance_2d_SI[:,0,0], label='velocity')
    ax.plot(omega_eV_p, absorbance_2d_SI_p[:,0,0], label='momentum')
    # ax.plot(omega_eV_p, a2, label='momentum')
    ax.set_xlabel(r"$\omega \ [\mathrm{eV}]$")
    ax.set_ylabel(r"Absorbance")
    ax.legend(loc='upper right',facecolor='#F2F2F2')
    ax.set_facecolor('#F2F2F2')
    plt.savefig("absorbance_2d.pdf")
    plt.show()
    
    