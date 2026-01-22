

import numpy as np
import os
import scipy as sc
from scipy.interpolate import RegularGridInterpolator


###########################
# Utility routines
###########################

def symmetrizeMatrixElements(cells, H, R):
    """if not unitary force unitarity"""
    cellMap = {}
    for i, c in enumerate(cells):
        cellMap[tuple(c)] = i
    Hn = np.empty(H.shape, dtype=complex)
    Rn = np.empty(R.shape, dtype=complex)
    for i, c in enumerate(cells):
        reflectedIndex = cellMap[(-c[0], -c[1], -c[2])]
        Hn[i] = 0.5 * (H[i] + H[reflectedIndex].conj().T)
        for d in range(3):
            Rn[i, :, :, d] = 0.5 * (R[i, :, :, d] + R[reflectedIndex, :, :, d].conj().T)
    return Hn, Rn

def angstrom_to_bohr(angstrom):
    meter = angstrom * sc.constants.angstrom
    bohr = meter / sc.constants.physical_constants['atomic unit of length'][0]
    return bohr 

def bohr_to_angstrom(bohr):
    meter = bohr * sc.constants.physical_constants['atomic unit of length'][0] 
    angstrom = meter / sc.constants.angstrom
    return angstrom

def eV_to_au(eV):
    return eV/sc.constants.physical_constants['Hartree energy in eV'][0]

def au_to_eV(au):
    return au * sc.constants.physical_constants['Hartree energy in eV'][0]
    

###########################
# Reading routines
###########################

""" Reads _tb file from wannier90 file. The units are eV and eV * Angstrom """
"""dont  use symmetrize"""
def read_tb(fname, symmetrize=False, onlyReal=False, onlyLattice=False):
    """Units in file:
            lattice-vectors: Angstrom
            H: eV
            S: no unit
            R: Angstrom
        Return:
            all atomic units

    """
    origFname = None
    for fn in [fname, fname + "_tb.dat", fname + ".dat"]:
        if os.path.exists(fn):
            origFname = fn
            break
    if origFname is None:
        print(f"read_tb: No matching file name found for '{fname}'")
        return
    with open(origFname, "r") as f:
        f.readline() # header
        lattice_ang = np.empty((3,3))
        for i in range(3):
            gv = f.readline() # grid vectors
            lattice_ang[i] = np.array([float(g) for g in gv.split()])
        lattice_au = angstrom_to_bohr(lattice_ang)
        if onlyLattice:
            return lattice_au
        numWann = int(f.readline())
        nR = int(f.readline())
        degeneracy = []
        while len(degeneracy) < nR:
            degeneracy += [float(s) for s in f.readline().split()]
        degeneracy = np.array(degeneracy)
        cells = np.empty((nR, 3), dtype=int)
        H = np.empty((nR, numWann, numWann), dtype=complex)
        S = np.empty((nR, numWann, numWann), dtype=complex)
        R = np.empty((nR, numWann, numWann, 3), dtype=complex)

        for ri in range(nR):
            f.readline()
            cells[ri, :] = np.array([int(s) for s in f.readline().split()])
            for a in range(numWann):
                for b in range(numWann):
                    sp = f.readline().split()
                    aS, bS = [int(s)-1 for s in sp[:2]]
                    Hreal, Himag = [float(s) for s in sp[2:4:]]
                    Sreal, Simag = [float(s) for s in sp[4:6:]]
                    if onlyReal:
                        Himag = 0
                        Simag = 0
                    H[ri, aS, bS] = Hreal + 1j * Himag
                    S[ri, aS, bS] = Sreal + 1j * Simag
                    
        # dipole transition
        for ri in range(nR):
            f.readline()
            rIndex = np.array([int(s) for s in f.readline().split()])
            assert np.all(rIndex == cells[ri] )
            for a in range(numWann):
                for b in range(numWann):
                    sp = f.readline().split()
                    aS, bS = [int(s)-1 for s in sp[:2]]
                    rReal = np.array([float(s) for s in sp[2::2]])
                    rImag = np.array([float(s) for s in sp[3::2]])
                    if onlyReal:
                        rImag = 0
                    R[ri, aS, bS] = rReal + 1j * rImag
    if symmetrize:
        H, R = symmetrizeMatrixElements(cells, H, R)
    H_au = eV_to_au(H)
    R_au = angstrom_to_bohr(R) 
    check_pos_op_properties(lattice_au=lattice_au, cells=cells, S=S, R_au=R_au)
    return lattice_au, cells, degeneracy, H_au, S, R_au 

def read_tb_orthogonal(fname, symmetrize=False, onlyReal=False, onlyLattice=False):
    """Units in file:
            lattice-vectors: Angstrom
            H: eV
            R: Angstrom
        Return:
            all atomic units

    """
    origFname = None
    for fn in [fname, fname + "_tb.dat", fname + ".dat"]:
        if os.path.exists(fn):
            origFname = fn
            break
    if origFname is None:
        print(f"read_tb: No matching file name found for '{fname}'")
        return
    with open(origFname, "r") as f:
        f.readline() # header
        lattice_ang = np.empty((3,3))
        for i in range(3):
            gv = f.readline() # grid vectors
            lattice_ang[i] = np.array([float(g) for g in gv.split()])
        lattice_au = angstrom_to_bohr(lattice_ang)
        if onlyLattice:
            return lattice_au
        numWann = int(f.readline())
        nR = int(f.readline())
        degeneracy = []
        while len(degeneracy) < nR:
            degeneracy += [float(s) for s in f.readline().split()]
        degeneracy = np.array(degeneracy)
        cells = np.empty((nR, 3), dtype=int)
        H = np.empty((nR, numWann, numWann), dtype=complex)
        S = np.empty((nR, numWann, numWann), dtype=complex)
        R = np.empty((nR, numWann, numWann, 3), dtype=complex)

        for ri in range(nR):
            f.readline()
            cells[ri, :] = np.array([int(s) for s in f.readline().split()])
            for a in range(numWann):
                for b in range(numWann):
                    sp = f.readline().split()
                    aS, bS = [int(s)-1 for s in sp[:2]]
                    Hreal, Himag = [float(s) for s in sp[2:4:]]
                    if onlyReal:
                        Himag = 0
                    H[ri, aS, bS] = Hreal + 1j * Himag
            S[ri,:,:] = np.eye(numWann)
                    
        # dipole transition
        for ri in range(nR):
            f.readline()
            rIndex = np.array([int(s) for s in f.readline().split()])
            assert np.all(rIndex == cells[ri] )
            for a in range(numWann):
                for b in range(numWann):
                    sp = f.readline().split()
                    aS, bS = [int(s)-1 for s in sp[:2]]
                    rReal = np.array([float(s) for s in sp[2::2]])
                    rImag = np.array([float(s) for s in sp[3::2]])
                    if onlyReal:
                        rImag = 0
                    R[ri, aS, bS] = rReal + 1j * rImag
    if symmetrize:
        H, R = symmetrizeMatrixElements(cells, H, R)
    H_au = eV_to_au(H)
    R_au = angstrom_to_bohr(R) 
    return lattice_au, cells, degeneracy, H_au, S, R_au 

def check_pos_op_properties(lattice_au, cells, S, R_au):
    """make sure that for every lattice point there is also the inverse in the list. 
    Also ensure that the pos. operator elements fulfill
        <m,0|r|n,R> = <n,0|r|m,-R> + R <n,0|m,-R> """
    print(f"Up to {np.max(np.abs(cells))} unit cell shifts")
    s = {tuple(v) for v in cells}
    if not all(tuple(-v) in s for v in cells):
        print("Real space lattice points not inversion symmetric")
    has_symmetry = True
    maxdiff = 0
    for i, cell in enumerate(cells):
        idx_minusR = np.where(np.all(cells == -cell, axis=1))[0]
        zero = R_au[i] - np.transpose(R_au[idx_minusR], axes=(0,2,1,3)).conj() - np.einsum('ba, b, rcd-> rcda', lattice_au, cell, np.transpose(S[idx_minusR], axes=(0,2,1)).conj())
        if np.max(np.abs(zero)) > maxdiff:
            maxdiff = np.max(np.abs(zero))
        if not np.allclose(zero, 0, atol=1e-9):
            has_symmetry = False
    if not has_symmetry:
        print(f"position operator does not fulfill symmetry requirement by up to {maxdiff}")


""" Reads wsvec file from wannier90 -- incorporating this improves interpolation """
def read_wsvec(fname):
    origFname = None
    for fn in [fname, fname + "_wsvec.dat", fname + ".dat"]:
        if os.path.exists(fn):
            origFname = fn
            break
    if origFname is None:
        print(f"read_wsvec: No matching file name found for '{fname}'")
        return
    with open(origFname, "r") as f:
        header = f.readline() # header
        use_ws_distance = 'use_ws_distance=.true.' in header
        wsvecs = []
        for baseVecLine in f:
            baseIndices = [int(s) for s in baseVecLine.split()]
            baseIndices[3] -= 1
            baseIndices[4] -= 1
            cnt = int(f.readline())
            T = []
            for i in range(cnt):
                T.append([int(s) for s in f.readline().split()])
            wsvecs.append([baseIndices, T])
    return use_ws_distance, wsvecs

""" Reads tb and wsvec file and adapts the matrix elements according to wsvec, such
    that the k-space interpolation can be applied directly.
    Units: H: eV, R : eV * A
"""
def read_wsvectb(seed, symmetrize=False, onlyReal=False):
    tb = read_tb(seed + "_tb.dat", onlyReal=onlyReal)
    if tb == None:
        return
    ws = read_wsvec(seed + "_wsvec.dat")
    if ws == None:
        return
    lattice, tb_cells, tb_degeneracy, tb_H, tb_R = tb
    cellToTb = {}
    for i, cell in enumerate(tb_cells):
        cellToTb[tuple(cell)] = i
    use_ws_distance, wsvecs = ws
    cells = []
    cellToId = {}
    cellCount = 0
    for baseCell, wsv in wsvecs:
        r1, r2, r3, m, n = baseCell
        for c in wsv:
            cc = tuple( c + b for c, b in zip(c, [r1, r2, r3] ) )
            if cc not in cellToId:
                cellToId[cc] = cellCount
                cellCount += 1
                cells.append(cc)
    numWann = tb_H.shape[1]
    H = np.zeros((cellCount, numWann, numWann), dtype=complex)
    R = np.zeros((cellCount, numWann, numWann, 3), dtype=complex)
    for baseCell, wsv in wsvecs:
        r1, r2, r3, m, n = baseCell
        tbId = cellToTb[(r1, r2, r3)]
        for c in wsv:
            cc = tuple( c + b for c, b in zip(c, [r1, r2, r3]) )
            cellId = cellToId[cc]
            H[cellId, m, n] += tb_H[tbId, m, n] / tb_degeneracy[tbId] / len(wsv)
            R[cellId, m, n] += tb_R[tbId, m, n] / tb_degeneracy[tbId] / len(wsv)
    if symmetrize:
        H, R = symmetrizeMatrixElements(cells, H, R)
    return lattice, cells, H, R

""" Reads unitary rotation matrices provided by wannier90 """
def read_u(fname):
    origFname = None
    for fn in [fname, fname + "_u.mat", fname + ".mat"]:
        if os.path.exists(fn):
            origFname = fn
            break
    if origFname is None:
        print(f"read_u: No matching file name found for '{fname}'")
        return
    with open(origFname, "r") as f:
        f.readline() # header
        num_kpts, num_wann, _ = [int(s) for s in f.readline().split()]
        U = np.empty((num_kpts, num_wann, num_wann), dtype=complex)
        kFrac = np.empty((num_kpts, 3))
        for ki in range(num_kpts):
            f.readline() # empty line
            kFrac[ki] = np.array([float(v) for v in f.readline().split()])
            for a in range(num_wann):
                for b in range(num_wann):
                    re, im = np.array([float(v) for v in f.readline().split()])
                    U[ki, b, a] = re + 1j * im
    return U, kFrac

###########################
# Interpolation routines
###########################


""" interpolates Hamiltonian to fractional k-point using the old interpolation scheme """
def Hk_degenerate(cells, degeneracies, Hr, kFrac):
    kr = 2 * np.pi * np.einsum("ab, ...b -> ...a", cells, kFrac)
    Hk = np.einsum("...a,abc -> ...bc",  np.exp(1j * kr), Hr / degeneracies[:, np.newaxis, np.newaxis])
    return Hk

""" interpolates dipole operator to fractional k-point using the old interpolation scheme """
def Rk_degenerate(cells, degeneracies, Rr, kFrac):
    kr = 2 * np.pi * np.einsum("ab, ...b -> ...a", cells, kFrac)
    Dk = np.einsum("...a,abcd->...bcd",  np.exp(1j * kr), Rr / degeneracies[:, np.newaxis, np.newaxis, np.newaxis])
    return Dk

""" interpolates Hamiltonian to fractional k-point using the new interpolation scheme
    Hint: data can be obtained by read_wsvectb
"""
def Hk(cells, Hr, kFrac):
    kr = 2 * np.pi * np.einsum("ab,...b ->...a", cells, kFrac)
    Hk = np.einsum("...a,abc->...bc",  np.exp(1j * kr), Hr)
    return Hk 

""" interpolates dipole operator to fractional k-point using the new interpolation scheme
    Hint: data can be obtained by read_wsvectb
"""
def Rk(cells, Rr, kFrac):
    kr = 2 * np.pi * np.einsum("ab, ...b -> ...a", cells, kFrac)
    Rk = np.einsum("...a,abcd->...bcd",  np.exp(1j * kr), Rr)
    return Rk


# def grad_H(cells, H, kFrac, lattice):
#     kr = 2 * np.pi * np.einsum("ab, ...b -> ...a", cells, kFrac)
#     R_cart = cells @ lattice
#     grad_H = 1j * 2 * np.pi * np.einsum("...a, abc, az -> ...bcz", np.exp(1j * kr), H, R_cart)
#     # grad_H /= cells.shape[0]
#     return grad_H

# def to_regular_grid(cells, H, S, D):
#     minx = np.min(cells[:,0])
#     miny = np.min(cells[:,1])
#     minz = np.min(cells[:,2])
#     maxx = np.max(cells[:,0])
#     maxy = np.max(cells[:,1])
#     maxz = np.max(cells[:,2])
#     Nx = maxx - minx + 1
#     Ny = maxy - miny + 1
#     Nz = maxz - minz + 1
#     numWann = np.shape(H)[1]
#     H_reg_grid = np.zeros((Nx, Ny, Nz, numWann, numWann), dtype=complex)
#     S_reg_grid = np.zeros((Nx, Ny, Nz, numWann, numWann), dtype=complex)
#     R_reg_grid = np.zeros((Nx, Ny, Nz, numWann, numWann, 3), dtype=complex)
#     lattice_offset = np.array([minx, miny, minz])
#     for i in range(np.shape(cells)[0]):
#         idx = cells[i,0] - minx
#         idy = cells[i,1] - miny
#         idz = cells[i,2] - minz
#         H_reg_grid[idx, idy, idz] = H[i]
#         S_reg_grid[idx, idy, idz] = S[i]
#         R_reg_grid[idx, idy, idz] = D[i]
#     return H_reg_grid, S_reg_grid, R_reg_grid, lattice_offset 

# def Hk_fft_interp(H_reg, lattice_offset, k_points, cells):
#     Nk = np.shape(H_reg)[:3]
#     grid_pos = [np.fft.fftshift(np.fft.fftfreq(n, d=1/n)) for n in Nk]
#     grid = [np.flip(n) for n in grid_pos] #fft has opposite sign in exponent
#     Hk_grid = np.fft.fftn(a=H_reg, axes=(0,1,2))/np.shape(cells)[0]
#     Hk_grid = np.fft.fftshift(x=Hk_grid, axes=(0,1,2))
#     interp = RegularGridInterpolator(points=grid, values=Hk_grid) # interpolation not necessary, shift by exp(ikR) in R-space
#     offset_phase = np.exp(2 * np.pi * 1j * np.dot(lattice_offset, k_points))
#     return interp(k_points) * offset_phase 
