import numpy as np
import os
from from_tb.utils import *

###########################
# Reading routines
###########################

""" Reads _tb file from wannier90 file. The units are eV and eV * Angstrom """
"""dont  use symmetrize"""
def read_tb(fname, onlyReal=False, onlyLattice=False):
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
    H_au = eV_to_au(H)
    R_au = angstrom_to_bohr(R) 
    check_pos_op_properties(lattice_au=lattice_au, cells=cells, S=S, R_au=R_au)
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

def grad_H_degenerate(cells, degeneracies, Hr, kFrac, lattice):
    kr = 2 * np.pi * np.einsum("ab, ...b -> ...a", cells, kFrac)
    R_cart = np.einsum('ab, cb->ca', lattice.T, cells) #lattice has lattice vectors as rows
    R_cart_alt = np.einsum('ab, bc-> ac', cells, lattice)
    assert np.allclose(R_cart, R_cart_alt)
    grad_H = 1j * np.einsum("...c, cmn, cx -> ...mnx", np.exp(1j * kr), Hr/ degeneracies[:,None, None], R_cart)
    return grad_H

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

