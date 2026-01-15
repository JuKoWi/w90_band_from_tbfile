from w90 import read_tb, eV_to_au, angstrom_to_bohr, au_to_eV
from extract_properties_tbfile import *
import numpy as np

def test_realspace_momentum(tb_file, shape_tuple):
    kPoints = k_grid(n_points=shape_tuple)
    lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb(tb_file)
    pk, Sk_orth, Hk_orth = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice)
    pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth, Sk_orth=Sk_orth) # (k, a, a, c)
    Nk, Nband, _, Ncoord = np.shape(pk_bloch)
    pk_bloch = np.reshape(pk_bloch, shape=(*shape_tuple, Nband, Nband, Ncoord)) #(kx, ky, kz, a, a, c)
    fft_momentum = np.fft.fftn(a=pk_bloch, axes=(0,1,2)) #(Rx, Ry, Rz, a, a, c)
    fft_momentum = np.transpose(fft_momentum, axes=(5, 0, 1, 2, 3, 4)) # (c, Rx, Ry, Rz, a, a)
    print(np.shape(fft_momentum))
    frobenius_p = np.linalg.matrix_norm(fft_momentum, ord='fro') # (c, Rx, Ry, Rz)
    print(np.shape(frobenius_p))
    return frobenius_p

def test_realspace_dipole(tb_file, shape_tuple):
    kPoints = k_grid(n_points=shape_tuple)
    lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb(tb_file)
    dk_orth, Sk_orth, Hk_orth = get_dipole(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice)
    dk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=dk_orth, Hk_orth=Hk_orth, Sk_orth=Sk_orth) # (k, a, a, c)
    Nk, Nband, _, Ncoord = np.shape(dk_bloch)
    dk_bloch = np.reshape(dk_bloch, shape=(*shape_tuple, Nband, Nband, Ncoord)) #(kx, ky, kz, a, a, c)
    fft_dipole = np.fft.fftn(a=dk_bloch, axes=(0,1,2)) #(Rx, Ry, Rz, a, a, c)
    fft_dipole = np.transpose(fft_dipole, axes=(5, 0, 1, 2, 3, 4)) # (c, Rx, Ry, Rz, a, a)
    print(np.shape(fft_dipole))
    frobenius_dipole = np.linalg.matrix_norm(fft_dipole, ord='fro') # (c, Rx, Ry, Rz)
    print(np.shape(frobenius_dipole))
    return frobenius_dipole

def fold_to_wignerseitz(Rextent):
    """for a certain k-grid, take the R-grid resulting from the FFT and 
    and shift them by an integer number of unit cells that map it as close as 
     possible to the coordinate origin """
    x = np.arange(stop=Rextent[0])
    y = np.arange(stop=Rextent[1])
    z = np.arange(stop=Rextent[2])
    xv, yv, zv = np.meshgrid(x,y,z) 
    xv = xv.flatten()
    yv = yv.flatten()
    zv = zv.flatten()
    Rpoints = np.column_stack((xv,yv,zv))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(Rpoints[:,0], Rpoints[:,1], Rpoints[:,2])
    # plt.show()


if __name__=="__main__":
    fold_to_wignerseitz(Rextent=(10,10,1))
    fileA = "seedname_mos2_full.dat"
    frobenius_px = test_realspace_dipole(tb_file=fileA, shape_tuple=(500,1,1))
    frobenius_py = test_realspace_dipole(tb_file=fileA, shape_tuple=(1,500,1))
    R = np.arange(stop=500)
    plt.plot(R, frobenius_px[0,:,0,0])
    plt.plot(R, frobenius_px[1,:,0,0])
    plt.show()
    plt.plot(R, frobenius_py[0,0,:,0])
    plt.plot(R, frobenius_py[1,0,:,0])
    plt.show()

