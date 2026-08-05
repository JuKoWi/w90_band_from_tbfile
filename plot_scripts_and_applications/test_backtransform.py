"""Analyze quality of Fourier interpolation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tb_calculations.backtransform_realspace import nonorthogonal_wannierfile, test_realspace_dipole
import matplotlib.pyplot as plt
import numpy as np
from tb_calculations.parse_and_FT import read_tb
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'
# dipole_frobenius_x = test_realspace_dipole(tb_file='./seedname_mos2/seedname_tb.dat', shape_tuple=(20,20,1))
# dipole_frobenius_y = test_realspace_dipole(tb_file='./seedname_mos2/seedname_tb.dat', shape_tuple=(10,10,1))
# nonorthogonal_wannierfile(filename_in='./seedname_mos2/normal/seedname_tb.dat', shape_tuple=(5,5,1))


def plot_mat_elem_decay(ncells, tb_file, fromFile, lattice):
    save_result = 'dipole_from.npy'
    if fromFile:
        dipole_frobenius_x = np.load(file=save_result)
    else:
        dipole_frobenius_x = test_realspace_dipole(tb_file=tb_file, shape_tuple=(ncells,ncells,1))
        np.save(file=save_result, arr=dipole_frobenius_x)
    cells = np.arange(stop=ncells)
    fig, axs = plt.subplots(ncols=2, figsize=(15,4.5))
    axs[0].semilogy(cells, dipole_frobenius_x[0,:,0,0], "o", label=r"$n_i = n_x$")
    axs[0].semilogy(cells, dipole_frobenius_x[0,0,:,0], "o", label=r"$n_i = n_y$")
    axs[0].set_xlabel(xlabel=r'$n_i$')
    axs[0].set_ylabel(ylabel=r'$|| d_x ||_F$ [a.u.]')
    axs[0].legend()
    axs[1].semilogy(cells, dipole_frobenius_x[1,:,0,0], "o", label=r"$n_i = n_x$")
    axs[1].semilogy(cells, dipole_frobenius_x[1,0,:,0], "o", label=r"$n_i = n_y$")
    axs[1].set_xlabel(xlabel=r'$n_i$')
    axs[1].set_ylabel(ylabel=r'$|| d_y ||_F$ [a.u.]')
    axs[1].legend()
    plt.savefig('dipole_frobenius.pdf')
    plt.show()

lattice_au, cells, degeneracy, H_au, S, R_au = read_tb(fname='./seedname_mos2/seedname_tb.dat')
plot_mat_elem_decay(ncells=20, tb_file='./seedname_mos2/seedname_tb.dat', fromFile=True, lattice=lattice_au)