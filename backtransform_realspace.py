from w90 import read_tb, eV_to_au, angstrom_to_bohr, au_to_eV, bohr_to_angstrom
from extract_properties_tbfile import *
from collections import Counter
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

def transform_hamiltonian(Hk, shape_tuple):
    """includes factor 1/N in backtransform because not included in R->k transform"""
    Nk, Nband, _ = np.shape(Hk)
    assert np.prod(shape_tuple) == Nk
    Hk_reshaped = np.reshape(Hk, shape=(*shape_tuple, Nband, Nband), order='C') #(kx, ky, kz, a, a)
    fft_hamiltonian = np.fft.fftn(a=Hk, axes=(0,1,2))/Nk  # (Rx, Ry, Rz, a, a)
    fft_hamiltonian = np.reshape(fft_hamiltonian, shape=(Nk, Nband, Nband), order='C')
    return fft_hamiltonian

def transform_dipole(dk, shape_tuple):
    """includes factor 1/N in backtransform because not included in R->k transform"""
    Nk, Nband, _, Ncoord = np.shape(dk)
    assert np.prod(shape_tuple) == Nk
    dk = np.reshape(dk, shape=(*shape_tuple, Nband, Nband, Ncoord), order='C') #(kx, ky, kz, a, a, c)
    fft_dipole = np.fft.fftn(a=dk, axes=(0,1,2))/Nk #(Rx, Ry, Rz, a, a, c)
    fft_dipole = np.reshape(fft_dipole, shape=(Nk, Nband, Nband, Ncoord), order='C')
    return fft_dipole

def fold_to_wignerseitz(Rextent, lattice):
    """for a certain k-grid, take the R-grid resulting from the FFT and 
    and shift them by an integer number of unit cells that map it as close as 
     possible to the coordinate origin 
     
     For a k-grid with the grid points defined as:
        k = n/N * vec{b}  with 0 <= n < N
    The R-points
        R = j vec{a} with 0 <= j < N
    are the points that give exactly i * n * j * 2 * pi /N in the exponent
    with the vectors b and a being defined following the solid-state-physics convention
     """
    x = np.arange(stop=Rextent[0])
    y = np.arange(stop=Rextent[1])
    z = np.arange(stop=Rextent[2])
    xv, yv, zv = np.meshgrid(x,y,z) 
    xv = xv.flatten(order='C')
    yv = yv.flatten(order='C')
    zv = zv.flatten(order='C')
    Rpoints = np.column_stack((xv,yv,zv))
    metric_tensor = np.einsum('ai,bi->ab', lattice, lattice)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(Rpoints[:,0], Rpoints[:,1], Rpoints[:,2])
    # plt.show()
    xcells = np.arange(start=-2, stop=3, step=1)
    ycells = np.arange(start=-2, stop=3, step=1)
    zcells = np.arange(start=-2, stop=3, step=1)
    xdir, ydir, zdir = np.meshgrid(xcells, ycells, zcells)
    image_grid = np.column_stack((xdir.flatten(order='C'), ydir.flatten(order='C'), zdir.flatten(order='C')))
    Rpoints_folded = np.zeros_like(Rpoints)
    for i, point in enumerate(Rpoints):
        images = point + image_grid * np.array([*Rextent])
        square_dist = np.einsum('ia,ab,ib->i', images, metric_tensor, images)
        Rpoints_folded[i] = images[np.argmin(square_dist)]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(Rpoints_folded[:,0], Rpoints_folded[:,1], Rpoints_folded[:,2])
    # plt.show()
    return Rpoints_folded

def get_equivalence_keys(Rpoints, Rextent):
    keys = Rpoints % np.asarray(Rextent)
    point_tuples = list(map(tuple, keys))
    counts = Counter(point_tuples)
    counts_per_elem = np.array([counts[x] for x in point_tuples])
    return keys, counts_per_elem

def write_wannierfile(Rpoints, degeneracy, lattice_au, Hr, posr, filename="seedname_orth.dat"):
    """write wannier90 file with hamiltonian Hr and position-operator posr for a
    set of lattice vectors Rpoints"""
    lattice_angstrom = bohr_to_angstrom(lattice_au)
    Hr = au_to_eV(Hr)
    posr = bohr_to_angstrom(posr)

    Norbs = np.shape(Hr)[-1]
    Nlattice = np.shape(Rpoints)[0]
    with open(filename, 'w') as f:
        f.write(str(np.datetime64('now'))+'\n')
        np.savetxt(f, w90.bohr_to_angstrom(lattice_angstrom))
        f.write(str(Norbs)+'\n')
        f.write(str(Nlattice)+'\n')
        for i in range(Nlattice):
            f.write(str(degeneracy[i])+' ')
            if (i+1) % 15 == 0:
                f.write('\n')
        f.write('\n')
    
def orthogonal_wannierfile(filename_in, shape_tuple):
    kPoints = k_grid(n_points=shape_tuple)
    lattice_au, cells, degeneracy, Hr, Sr, Rr = w90.read_tb(filename_in)
    dk_orth, Sk_orth, Hk_orth = get_dipole(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice_au) # (k,a,a), (k,a,a), (k,a,a,c)
    Hr = transform_hamiltonian(Hk=Hk_orth, shape_tuple=shape_tuple)
    posr = transform_dipole(dk=dk_orth, shape_tuple=shape_tuple)
    Rpoints_WScell = fold_to_wignerseitz(Rextent=shape_tuple, lattice=lattice_au)
    keys, counts_per_point = get_equivalence_keys(Rpoints=Rpoints_WScell, Rextent=shape_tuple)
    write_wannierfile(Hr=Hr, posr=posr, Rpoints=Rpoints_WScell, degeneracy=counts_per_point, lattice_au=lattice_au)

if __name__=="__main__":
    fileA = "seedname_mos2_full.dat"
    orthogonal_wannierfile(filename_in=fileA, shape_tuple=(10,10,1))
    # frobenius_px = test_realspace_dipole(tb_file=fileA, shape_tuple=(500,1,1))
    # frobenius_py = test_realspace_dipole(tb_file=fileA, shape_tuple=(1,500,1))
    # R = np.arange(stop=500)
    # plt.plot(R, frobenius_px[0,:,0,0])
    # plt.plot(R, frobenius_px[1,:,0,0])
    # plt.show()
    # plt.plot(R, frobenius_py[0,0,:,0])
    # plt.plot(R, frobenius_py[1,0,:,0])
    # plt.show()

