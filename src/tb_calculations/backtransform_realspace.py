"""Functions to convert quantities back to real space basis (tight binding file), 
to generate orthogonal tight binding file from non-orthogonal tight binding file 
"""

from tb_calculations.parse_and_FT import read_tb 
from tb_calculations.extract_observables import *
from tb_calculations.utils import k_grid, bohr_to_angstrom, au_to_eV
from collections import Counter
import numpy as np
import time
import sys

def test_realspace_momentum(tb_file, shape_tuple):
    kPoints = k_grid(n_points=shape_tuple)
    lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(tb_file)
    pk, Sk_orth, Hk_orth = get_momentum_orth(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth, Sk_orth=Sk_orth) # (k, a, a, c)
    Nk, Nband, _, Ncoord = np.shape(pk_bloch)
    pk_bloch = np.reshape(pk_bloch, shape=(*shape_tuple, Nband, Nband, Ncoord)) #(kx, ky, kz, a, a, c)
    fft_momentum = np.fft.fftn(a=pk_bloch, axes=(0,1,2)) #(Rx, Ry, Rz, a, a, c)
    fft_momentum = np.transpose(fft_momentum, axes=(5, 0, 1, 2, 3, 4)) # (c, Rx, Ry, Rz, a, a)
    frobenius_p = np.linalg.matrix_norm(fft_momentum, ord='fro') # (c, Rx, Ry, Rz)
    return frobenius_p

def test_realspace_dipole(tb_file, shape_tuple):
    kPoints = k_grid(n_points=shape_tuple)
    lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(tb_file)
    dk_orth, Sk_orth, Hk_orth = get_dipole_orth(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    # dk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=dk_orth, Hk_orth=Hk_orth, Sk_orth=Sk_orth) # (k, a, a, c)
    Nk, Nband, _, Ncoord = np.shape(dk_orth)
    dk_orth = np.reshape(dk_orth, shape=(*shape_tuple, Nband, Nband, Ncoord)) #(kx, ky, kz, a, a, c)
    fft_dipole = np.fft.fftn(a=dk_orth, axes=(0,1,2))/Nk #(Rx, Ry, Rz, a, a, c)
    fft_dipole = np.transpose(fft_dipole, axes=(5, 0, 1, 2, 3, 4)) # (c, Rx, Ry, Rz, a, a)
    frobenius_dipole = np.linalg.matrix_norm(fft_dipole, ord='fro') # (c, Rx, Ry, Rz)
    return frobenius_dipole

def transform_hamiltonian(Hk, shape_tuple):
    """includes factor 1/N in backtransform because not included in R->k transform"""
    Nk, Nband, _ = np.shape(Hk)
    if not (np.prod(shape_tuple) == Nk):
        print("Mismatch in shape tuple and actual shape of array")
    Hk_reshaped = np.reshape(Hk, shape=(*shape_tuple, Nband, Nband), order='C') #(kx, ky, kz, a, a)
    fft_hamiltonian = np.fft.fftn(a=Hk_reshaped, axes=(0,1,2))/Nk  # (Rx, Ry, Rz, a, a)
    fft_hamiltonian = np.reshape(fft_hamiltonian, shape=(Nk, Nband, Nband), order='C')
    return fft_hamiltonian

def transform_dipole(dk, shape_tuple):
    """includes factor 1/N in backtransform because not included in R->k transform"""
    Nk, Nband, _, Ncoord = np.shape(dk)
    assert np.prod(shape_tuple) == Nk
    dk = np.reshape(dk, shape=(*shape_tuple, Nband, Nband, Ncoord)) #(kx, ky, kz, a, a, c)
    fft_dipole = np.fft.fftn(a=dk, axes=(0,1,2))/Nk #(Rx, Ry, Rz, a, a, c)
    fft_dipole = np.reshape(fft_dipole, shape=(Nk, Nband, Nband, Ncoord) )
    return fft_dipole

def fold_to_wignerseitz(Rextent, lattice, fft_matrices):
    """for a certain k-grid, take the R-grid resulting from the FFT and 
    and shift them by an integer number of unit cells that map it as close as 
     possible to the coordinate origin 
     
     For a k-grid with the grid points defined as:
        k = n/N * vec{b}  with 0 <= n < N
    The R-points
        R = j vec{a} with 0 <= j < N
    are the points that give exactly i * n * j * 2 * pi /N in the exponent
    with the vectors b and a being defined following the solid-state-physics convention
    If there are several possible shifts by whole supercells that give the same minimal distance to the origin 
    use all of them and mark as degeneracy
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
    Rpoints_folded = []
    ndeg = []
    repeats = np.zeros(shape=len(Rpoints), dtype=int)
    for i, point in enumerate(Rpoints):
        images = point + image_grid * np.array([*Rextent])
        square_dist = np.einsum('ia,ab,ib->i', images, metric_tensor, images)
        min_dist = np.min(square_dist)
        min_dist_images = np.isclose(square_dist, min_dist, atol=1e-4)
        degeneracy = np.sum(min_dist_images)
        # if degeneracy > 1:
        #     print(images[min_dist_images])
        repeats[i] = degeneracy
        degeneracy = [degeneracy] * degeneracy
        ndeg.extend(degeneracy)
        min_dist_images = images[min_dist_images]
        Rpoints_folded.extend(min_dist_images)
    Rpoints_folded = np.array(Rpoints_folded)
    ndeg = np.array(ndeg)
    fft_matrices_folded = []
    for i, mat in enumerate(fft_matrices):
        fft_matrices_folded.append(np.repeat(mat, axis=0, repeats=repeats)) 
        assert np.shape(fft_matrices_folded[i])[0] == np.shape(Rpoints_folded)[0]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(Rpoints_folded[:,0], Rpoints_folded[:,1], Rpoints_folded[:,2])
    # plt.show()
    return Rpoints_folded, ndeg, fft_matrices_folded 

def write_wannierfile(Rpoints, degeneracy, lattice_au, Hr, posr, Sr=None, filename="seedname_orth.dat"):
    """write wannier90 file with hamiltonian Hr and position-operator posr for a
    set of lattice vectors Rpoints"""
    lattice_angstrom = bohr_to_angstrom(lattice_au)
    Hr = au_to_eV(Hr)
    posr = bohr_to_angstrom(posr)

    Norbs = np.shape(Hr)[-1]
    Nlattice = np.shape(Rpoints)[0]
    with open(filename, 'w') as f:
        f.write(str(np.datetime64('now'))+'\n')
        np.savetxt(f, lattice_angstrom)
        f.write(str(Norbs)+'\n')
        f.write(str(Nlattice)+'\n')
        for i in range(Nlattice):
            f.write(str(degeneracy[i])+' ')
            if (i+1) % 15 == 0:
                f.write('\n')
        f.write('\n')
        for n, Rvec in enumerate(Rpoints):
            f.write('\n')
            f.write(str(Rvec[0]) + ' ' + str(Rvec[1]) + ' ' + str(Rvec[2]) + '\n')
            A = np.real(Hr[n])
            B = np.imag(Hr[n])
            if Sr is not None:
                C = np.real(Sr[n])
                D = np.imag(Sr[n])
            for i in range(Norbs):
                for j in range(Norbs):
                    if Sr is None:
                        print(f"{i+1} {j+1}\t{A[i,j]:.18e}\t{B[i,j]:.18e}", file=f)
                    else:
                        print(f"{i+1} {j+1}\t{A[i,j]:.18e}\t{B[i,j]:.18e}\t{C[i,j]:.18e}\t{D[i,j]:.18e}", file=f)
        for n,Rvec in enumerate(Rpoints):
            f.write('\n')
            f.write(str(Rvec[0]) + ' ' + str(Rvec[1]) + ' ' + str(Rvec[2]) + '\n')
            xre = np.real(posr[n,:,:,0])
            xim = np.imag(posr[n,:,:,0])
            yre = np.real(posr[n,:,:,1])
            yim = np.imag(posr[n,:,:,1])
            zre = np.real(posr[n,:,:,2])
            zim = np.imag(posr[n,:,:,2])
            for i in range(Norbs):
                for j in range(Norbs):
                    print(f"{i+1} {j+1}\t{xre[i,j]:.18e}\t{xim[i,j]:.18e}\t{yre[i,j]:.18e}\t{yim[i,j]:.18e}\t{zre[i,j]:.18e}\t{zim[i,j]:.18e}", file=f)
    
def orthogonal_wannierfile(filename_in, shape_tuple):
    start = time.time()
    kPoints = k_grid(n_points=shape_tuple)
    lattice_au, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(filename_in)
    dk_orth, Sk_orth, Hk_orth = get_dipole_orth(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice_au, degeneracies=degeneracies) # (k,a,a), (k,a,a), (k,a,a,c)
    Hr = transform_hamiltonian(Hk=Hk_orth, shape_tuple=shape_tuple)
    posr = transform_dipole(dk=dk_orth, shape_tuple=shape_tuple)
    fftmatrices = [Hr, posr]
    Rpoints_folded, ndeg, fftmatrices_folded = fold_to_wignerseitz(Rextent=shape_tuple, lattice=lattice_au, fft_matrices=fftmatrices)
    Hr_folded = fftmatrices_folded[0]
    posr_folded = fftmatrices_folded[1]
    print(f'maximal number of unit cells in any direction: {np.max(Rpoints_folded)}')
    print('start writing')
    write_wannierfile(Hr=Hr_folded, posr=posr_folded, Rpoints=Rpoints_folded, degeneracy=ndeg, lattice_au=lattice_au)
    stop = time.time()
    print(f"Function took {stop-start} s to execute")

def nonorthogonal_wannierfile(filename_in, shape_tuple):
    start = time.time()
    kPoints = k_grid(n_points=shape_tuple)
    lattice_au, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(filename_in)
    dk_orth, Sk_orth, Hk_orth = get_dipole_orth(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice_au, degeneracies=degeneracies) # (k,a,a), (k,a,a), (k,a,a,c)
    # dk_orth = make_momentum_hermitian(dk_orth)
    Hr = transform_hamiltonian(Hk=Hk_orth, shape_tuple=shape_tuple)
    Sr = transform_hamiltonian(Hk=Sk_orth, shape_tuple=shape_tuple)
    posr = transform_dipole(dk=dk_orth, shape_tuple=shape_tuple)
    fftmatrices = [Hr, posr, Sr]
    Rpoints_folded, ndeg, fftmatrices_folded = fold_to_wignerseitz(Rextent=shape_tuple, lattice=lattice_au, fft_matrices=fftmatrices)
    Hr_folded = fftmatrices_folded[0]
    posr_folded = fftmatrices_folded[1]
    Sr_folded = fftmatrices_folded[2]
    print(f'maximal number of unit cells in any direction: {np.max(Rpoints_folded)}')
    print('start writing')
    write_wannierfile(Hr=Hr_folded, posr=posr_folded, Rpoints=Rpoints_folded, degeneracy=ndeg, lattice_au=lattice_au, Sr=Sr_folded, filename="seedname_nonorth.dat")
    stop = time.time()
    print(f"Function took {stop-start} s to execute")

def compare_intermediate_steps(filename_A, filename_B, shape_tuple):
    """seems to only give the same results with 20x20 grid and above"""
    kPoints = k_grid(n_points=shape_tuple)
    latticeA, cellsA, degeneraciesA, HrA, SrA, RrA = parse_and_FT.read_tb(filename_A)
    latticeB, cellsB, degeneraciesB, HrB, SrB, RrB = parse_and_FT.read_tb(filename_B)
    dk_orthA, Sk_orthA, Hk_orthA = get_dipole_orth(Hr=HrA, Sr=SrA, Rr=RrA, kPoints=kPoints, cells=cellsA, lattice=latticeA, degeneracies=degeneraciesA)
    dk_orthB, Sk_orthB, Hk_orthB = get_dipole_orth(Hr=HrB, Sr=SrB, Rr=RrB, kPoints=kPoints, cells=cellsB, lattice=latticeB, degeneracies=degeneraciesB)
    print(np.allclose(Sk_orthA, Sk_orthB))
    print(np.allclose(dk_orthA, dk_orthB))
    print(np.allclose(Hk_orthA, Hk_orthB))
    pA, Sk_orthA, Hk_orthA = get_momentum_orth(Hr=HrA, Sr=SrA, Rr=RrA, degeneracies=degeneraciesA, kPoints=kPoints, cells=cellsA, lattice=latticeA)
    pB, Sk_orthB, Hk_orthB = get_momentum_orth(Hr=HrB, Sr=SrB, Rr=RrB, degeneracies=degeneraciesB, kPoints=kPoints, cells=cellsB, lattice=latticeB)
    print(np.allclose(pA, pB, atol=1e-8))
    print(np.allclose(Sk_orthA, Sk_orthB))
    print(np.allclose(Hk_orthA, Hk_orthB))

