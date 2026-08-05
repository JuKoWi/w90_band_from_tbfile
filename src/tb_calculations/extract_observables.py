"""
Functions for calculations in reciprocal space including band structure calculation 
and calculation of absorption spectrum
"""

import tb_calculations.parse_and_FT as parse_and_FT
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import time
from tb_calculations.utils import parsePath, MoS2_labelToK, get_rec_lattice, fermi_dirac, fermi_dirac_dE, check_vector_hermitian
plt.rcParams.update({'font.size': 30})
plt.rcParams['savefig.bbox'] = 'tight'

def bandstructure_orth_basis(lattice, cells, degeneracies, Hr, Sr, Rr):
    """gives all relevant quantities in an orthogonal basis to check for band structure"""
    segments, labels = parsePath("GMKG", lattice=lattice, labelToK=MoS2_labelToK)
    bands_orth = []
    d_orth = []
    H_orth = []
    S_orth = []
    for i, (kPoints, relPos) in enumerate(segments):
        Sk_orth, Hk_orth, dk_orth, grad_H = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints, degeneracies=degeneracies)
        vals, vecs = sc.linalg.eigh(Hk_orth)
        bands_orth.append(np.real(vals))
        H_orth.append(Hk_orth)
        S_orth.append(Sk_orth)
        d_orth.append(dk_orth)
    return segments, labels, bands_orth, H_orth, S_orth, d_orth 


def orthogonalize(lattice, cells, degeneracies, Hr, Sr, Rr, kPoints):
    """converts all matrices to orthogonal basis"""
    Hk = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints, degeneracies=degeneracies)
    Sk = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints, degeneracies=degeneracies)
    Rk = parse_and_FT.Rk_degenerate(cells=cells, Rr=Rr, kFrac=kPoints, degeneracies=degeneracies) # not hermitian!
    A1 = np.all(sc.linalg.ishermitian(Sk, atol=1e-12))
    B1 = np.all(sc.linalg.ishermitian(Hk, atol=1e-12))
    if not (A1 and B1):
        print("H and S not hermitian")
    S_inv_sqrt = diagonalization_inv_sqrt(Sk)
    if not np.all(sc.linalg.ishermitian(S_inv_sqrt, atol=1e-12)):
        print("Orthogonalization matrix not hermitian")
    grad_S_inv_sqrt, grad_H = grad_Hk_S_invsqrt(Hr=Hr, Sr=Sr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    if not np.allclose(grad_H, alternative_grad_Hk(Hr=Hr, Sr=Sr, kPoints=kPoints, lattice=lattice, cells=cells, degeneracies=degeneracies)):
        print(f"Two different ways of gradH in orth basis give different results")
    gradS_inv_sqrt_dagger = np.transpose(grad_S_inv_sqrt.conj(), axes=(0,2,1,3))
    if not check_vector_hermitian(pk=gradS_inv_sqrt_dagger, atol=1e-11): #only hermitian with atol = 1e-9
        print("grad S-1/2 not hermitian")
    Hk_orth = S_inv_sqrt @ Hk @ S_inv_sqrt
    Sk_orth = S_inv_sqrt @ Sk @ S_inv_sqrt
    A2 = np.all(sc.linalg.ishermitian(Sk_orth, atol=1e-12))
    B2 = np.all(sc.linalg.ishermitian(Hk_orth, atol=1e-12))
    if not (A2 and B2):
        print("S and H in orthogonal basis are not hermitian")
    S_inv_sqrt_dagger = np.transpose(S_inv_sqrt, axes=(0,2,1)).conj() #should not be necessary but maybe for numerical stability?
    dk_orth = -1j * np.einsum('kabz, kbc, kcd -> kadz', gradS_inv_sqrt_dagger, Sk, S_inv_sqrt) + np.einsum('kab,kbcz,kcd->kadz',S_inv_sqrt_dagger, Rk, S_inv_sqrt)
    hermitian_tol = np.max(np.abs(dk_orth - np.transpose(dk_orth, axes=(0,2,1,3)).conj()))
    if not check_vector_hermitian(dk_orth, atol=1e-9):
        print(f"Berry connection not hermitian by {hermitian_tol}")
    if not check_vector_hermitian(grad_H, atol=1e-9):
        print("gradient of H not hermitian")
    return Sk_orth, Hk_orth, dk_orth, grad_H

"""Diagonalization """

def diagonalization_inv_sqrt(Sk):
    vals, vecs = sc.linalg.eigh(Sk)
    U = vecs
    Udagger = np.transpose(vecs, axes=(0,2,1)).conj()
    minus_half = 1/np.sqrt(vals) 
    N,M = vals.shape
    diags_minushalf = np.zeros((N,M,M), dtype=complex)
    idx = np.arange(M)
    diags_minushalf[:,idx, idx] = minus_half
    S_minushalf = np.einsum('kab, kbc, kcd -> kad', U, diags_minushalf, Udagger) # backtransform from eigenbasis -> first U, then Udagger
    return S_minushalf

# def newton_schulz(Sk, iters=20):
#     k, n, _ = Sk.shape
#     I = np.eye(n)[None, :, :]          # shape (1, n, n)
#     I = np.broadcast_to(I, (k, n, n))  # shape (k, n, n)
#     normS = np.linalg.norm(Sk, axis=(1, 2), keepdims=True)
#     X = I / np.sqrt(normS)
#     for _ in range(iters):
#         X2 = X @ X
#         X = 0.5 * X @ (3*I - Sk @ X2)
#     return X

"""Partial derivatives"""

def partial_Hk(Hr, kPoints, cells, component, degeneracies):
    """This function does not include a change of basis"""
    stencil_shift_factor = 1e-3 # smaller stencil shift does not make sense
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    H_plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_plus, degeneracies=degeneracies) 
    H_minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_minus, degeneracies=degeneracies)
    H_2plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    H_2minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    derivative = (8 * H_plus - 8 * H_minus + H_2minus - H_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    if not np.all(sc.linalg.ishermitian(derivative, atol=1e-12)):
        print("partial Hk not hermitian")
    return derivative

def partial_Hk_S_invsqrt(Hr, Sr, kPoints, cells, component, degeneracies):
    stencil_shift_factor = 1e-3 # smaller stencil shift does not make sense
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    H_plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_plus, degeneracies=degeneracies) 
    H_minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_minus, degeneracies=degeneracies)
    H_2plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    H_2minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    S_plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_plus, degeneracies=degeneracies)
    S_minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_minus, degeneracies=degeneracies)
    S_2plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    S_2minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    Shalf_plus = diagonalization_inv_sqrt(S_plus)
    Shalf_minus = diagonalization_inv_sqrt(S_minus)
    Shalf_2plus = diagonalization_inv_sqrt(S_2plus)
    Shalf_2minus = diagonalization_inv_sqrt(S_2minus)
    Horth_plus = np.einsum('kab, kbc, kcd -> kad', Shalf_plus, H_plus, Shalf_plus)
    Horth_minus = np.einsum('kab, kbc, kcd -> kad', Shalf_minus, H_minus, Shalf_minus)
    Horth_2plus = np.einsum('kab, kbc, kcd -> kad', Shalf_2plus, H_2plus, Shalf_2plus)
    Horth_2minus = np.einsum('kab, kbc, kcd -> kad', Shalf_2minus, H_2minus, Shalf_2minus)
    derivativeH = (8 * Horth_plus - 8 * Horth_minus + Horth_2minus - Horth_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    derivativeS_inv_sqrt = (8 * Shalf_plus - 8 * Shalf_minus + Shalf_2minus - Shalf_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    return derivativeS_inv_sqrt, derivativeH

def partial_Sminushalf(Sr, kPoints, cells, degeneracies, component):
    stencil_shift_factor = 1e-3 #smaller stencil shift does not make sense
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    S_plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_plus, degeneracies=degeneracies)
    S_minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_minus, degeneracies=degeneracies)
    S_2plus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    S_2minus = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    Shalf_plus = diagonalization_inv_sqrt(S_plus)
    Shalf_minus = diagonalization_inv_sqrt(S_minus)
    Shalf_2plus = diagonalization_inv_sqrt(S_2plus)
    Shalf_2minus = diagonalization_inv_sqrt(S_2minus)
    A = np.all(sc.linalg.ishermitian(Shalf_plus, atol=1e-13))
    B = np.all(sc.linalg.ishermitian(Shalf_minus, atol=1e-13))
    C = np.all(sc.linalg.ishermitian(Shalf_2plus, atol=1e-13))
    D = np.all(sc.linalg.ishermitian(Shalf_2minus, atol=1e-13))
    if not (A and B and C and D):
        print("not all S-1/2 stencil points hermitian")
    derivative = (8 * Shalf_plus - 8 * Shalf_minus + Shalf_2minus - Shalf_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    if not np.all(sc.linalg.ishermitian(derivative, atol=1e-12)):
        print("partial derivative of S-1/2 not hermitian")
    return derivative

"""Assemble gradients"""

def alternative_grad_Hk(Hr, Sr, lattice, kPoints, cells, degeneracies):
    """Alternative way to calculate grad H in orthogonal basis"""
    Sk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    Hk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    S_inv_sqrt = diagonalization_inv_sqrt(Sk=Sk)
    S_inv_sqrt_grad = gradient_S_inv_sqrt(Sr, kPoints=kPoints, lattice=lattice, cells=cells, degeneracies=degeneracies)
    gradHk = gradient_H_atomic(Hr=Hr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    analytical_gradHk = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr,kFrac=kPoints, lattice=lattice)
    assert np.allclose(analytical_gradHk, gradHk)
    A = np.einsum('kabx, kbc, kcd -> kadx', S_inv_sqrt_grad, Hk, S_inv_sqrt)
    B = np.einsum('kab,kbcx,kcd->kadx', S_inv_sqrt, gradHk, S_inv_sqrt)
    C = np.einsum('kab,kbc,kcdx->kadx', S_inv_sqrt, Hk, S_inv_sqrt_grad)
    return A+B+C

def grad_Hk_S_invsqrt(Hr, Sr, kPoints, cells, lattice, degeneracies):
    """calculate gradient of hamiltonian in orthogonal basis and grad S_inv_sqrt"""
    Sa, Ha = partial_Hk_S_invsqrt(Hr=Hr, Sr=Sr, kPoints=kPoints, cells=cells, component=0, degeneracies=degeneracies)
    Sb, Hb = partial_Hk_S_invsqrt(Hr=Hr, Sr=Sr, kPoints=kPoints, cells=cells, component=1, degeneracies=degeneracies)
    Sc, Hc = partial_Hk_S_invsqrt(Hr=Hr, Sr=Sr, kPoints=kPoints, cells=cells, component=2, degeneracies=degeneracies)
    rec_lat_inv = lattice/(2 * np.pi) # equivalent to inverse of reciprocal lattice matrix. lattice has lattice vectors as rows. 
    grad_H = np.stack((Ha, Hb, Hc), axis=-1)
    grad_H = np.einsum('kabc, cd ->kabd', grad_H, rec_lat_inv)
    grad_S_inv_sqrt = np.stack((Sa, Sb, Sc), axis=-1)
    grad_S_inv_sqrt = np.einsum('kabc, cd ->kabd', grad_S_inv_sqrt, rec_lat_inv)
    return grad_S_inv_sqrt, grad_H

def gradient_H_atomic(Hr, kPoints, cells, lattice, degeneracies):
    """Returns array of shape (kpoints, basis, basis, components)"""
    Ha = partial_Hk(Hr=Hr, kPoints=kPoints, cells=cells, component=0, degeneracies=degeneracies)
    Hb = partial_Hk(Hr=Hr, kPoints=kPoints, cells=cells, component=1, degeneracies=degeneracies)
    Hc = partial_Hk(Hr=Hr, kPoints=kPoints, cells=cells, component=2, degeneracies=degeneracies)
    rec_lat_inv= lattice/ (2 * np.pi) # equivalent to inverse of reciprocal lattice matrix. lattice has lattice vectors as rows
    gradH = np.stack((Ha, Hb, Hc), axis=-1)
    gradH = np.einsum('kabc, cd ->kabd', gradH, rec_lat_inv)
    return gradH

def gradient_S_inv_sqrt(Sr, kPoints, lattice, cells, degeneracies):
    """Returns array of shape (kpoints, basis, basis, components)"""
    rec_lat_inv= lattice/ (2 * np.pi) # equivalent to inverse of reciprocal lattice matrix. lattice has lattice vectors as rows
    Sa = partial_Sminushalf(Sr=Sr, kPoints=kPoints, cells=cells, component=0, degeneracies=degeneracies)
    Sb = partial_Sminushalf(Sr=Sr, kPoints=kPoints, cells=cells, component=1, degeneracies=degeneracies)
    Sc = partial_Sminushalf(Sr=Sr, kPoints=kPoints, cells=cells, component=2, degeneracies=degeneracies)
    grad_S_inv_sqrt = np.stack((Sa, Sb, Sc), axis=-1)
    grad_S_inv_sqrt = np.einsum('kabc, cd ->kabd', grad_S_inv_sqrt, rec_lat_inv)
    return grad_S_inv_sqrt

"""Calculate dipole in different bases"""

def get_dipole_orth(Hr, Sr, Rr, kPoints, cells, lattice, degeneracies):
    Sk_orth, Hk_orth, dk_orth, grad_H = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints, degeneracies=degeneracies)
    return dk_orth, Sk_orth, Hk_orth

def get_dipole_atomic(Rr, Sr, kPoints, cells, lattice, degeneracies):
    gradS = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints, lattice=lattice)
    gradS_alt = gradient_H_atomic(Hr=Sr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    assert np.allclose(gradS, gradS_alt)
    Rk = parse_and_FT.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kPoints)
    return Rk + 1j* gradS

"""Calculate momentum in different bases"""

def get_momentum_orth(Hr, Sr, Rr, degeneracies, kPoints, cells, lattice):
    """Calculate momentum in orthogonal, non-Bloch basis"""
    Sk_orth, Hk_orth, dk_orth, grad_H = orthogonalize(lattice=lattice, cells=cells, degeneracies=degeneracies, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints)
    if not check_vector_hermitian(pk=grad_H, atol=1e-10):
        print("gradient H not hermitian")
    commutator = np.einsum('kabz, kbc->kacz', dk_orth, Hk_orth) - np.einsum('kab, kbcz -> kacz', Hk_orth, dk_orth)
    if not check_vector_hermitian(pk=1j*commutator, atol=1e-8):
        print("i * [d, H] is not hermitian")
    p = +1j * commutator + grad_H
    return p, Sk_orth, Hk_orth

def get_momentum_bloch(Hr, Sr, Rr, degeneracies, kPoints, cells, lattice):
    """Calculate momentum directly in Bloch basis according to formula 21 from 10.21468/SciPostPhysCore.6.1.002"""
    Sk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    Rk = parse_and_FT.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kPoints)
    Hk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    gradS_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints, lattice=lattice)
    gradH_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints, lattice=lattice)
    Ak_atomic = Rk + 1j * gradS_atomic
    eigvals, U = sc.linalg.eigh(Hk, Sk)
    U_dagger = np.transpose(U, axes=(0,2,1)).conj()
    term1 = np.einsum("kab, kbcz, kcd -> kadz", U_dagger, gradH_atomic, U)
    term2 = 1j * np.einsum("kab, kbcz, kcd, ka -> kadz", U_dagger, Ak_atomic, U, eigvals)
    term3 = -1j * np.einsum('kab, kbcz, kcd, kd -> kadz', U_dagger, np.transpose(Ak_atomic, axes=(0,2,1,3)).conj(), U, eigvals)
    if not check_vector_hermitian(pk= term1 + term2 + term3, atol=1e-8):
        print(f"Momentum after Paredes-formula is not hermitian")
    return term1 + term2 + term3

def get_momentum_bloch_lee(Hr, Sr, Rr, degeneracies, kPoints, cells, lattice):
    """Calculate momentum directly in Bloch basis according to formula 7 from 10.1103/PhysRevB.98.115115"""
    Sk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    Rk = parse_and_FT.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kPoints)
    Hk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    gradH_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints, lattice=lattice)
    gradS_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints, lattice=lattice)
    eigvals, U = sc.linalg.eigh(Hk, Sk)
    U_dagger = np.transpose(U, axes=(0,2,1)).conj()
    term1 = np.einsum('kab, kbcz, kcd -> kadz', U_dagger, gradH_atomic, U)
    term2 = -np.einsum('kab, kbcz, kcd, ka -> kadz', U_dagger, gradS_atomic, U, eigvals)
    term3 = 1j * np.einsum('kab, kbcz, kcd, ka -> kadz', U_dagger, Rk, U, eigvals)
    term4 = -1j * np.einsum('kab, kbcz, kcd, kd -> kadz', U_dagger, Rk, U, eigvals)
    if not check_vector_hermitian(pk= term1 + term2 + term3 + term4, atol=1e-7):
        print(f"Momentum after Lee-formula is not hermitian")
    return term1 + term2 + term3 + term4

def get_velocity_atomic(Hr, Sr, Rr, degeneracies, kFrac, cells, lattice):
    """get velocity elements directly in atomic basis"""
    Sk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kFrac)
    Rk = parse_and_FT.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kFrac)
    Hk = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kFrac)
    gradH_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kFrac, lattice=lattice)
    gradS_atomic = parse_and_FT.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kFrac, lattice=lattice)
    gradS_invsqrt = gradient_S_inv_sqrt(Sr=Sr, kPoints=kFrac, lattice=lattice, cells=cells, degeneracies=degeneracies) 
    berry_conn = Rk + 1j* gradS_atomic
    Sinv_sqrt = diagonalization_inv_sqrt(Sk=Sk)
    Sinv = np.einsum('kab, kbc-> kac', Sinv_sqrt, Sinv_sqrt)
    gradSinv = np.einsum('kabz, kbc->kacz', gradS_invsqrt, Sinv_sqrt) + np.einsum('kab, kbcz -> kacz', Sinv_sqrt, gradS_invsqrt)
    term1 = 1j* np.einsum('kab, kbc, kcdz -> kadz', Hk, Sinv,berry_conn)
    term2 = np.einsum('kab, kbcz, kcd -> kadz', Sk, gradSinv, Hk)
    term3 = gradH_atomic
    term4 = -1j * np.einsum('kabz, kbc, kcd->kadz',berry_conn,Sinv,Hk)
    if not check_vector_hermitian(pk= term1 + term2 + term3 + term4, atol=1e-8):
        print(f"Velocity in nonorth basis is not hermitian")
    return term1 + term2 + term3 + term4


def get_momentum_atomic(Hr, Sr, pr, degeneracies, kFrac, cells, lattice):
    return -1j *parse_and_FT.Rk_degenerate(cells=cells, Rr=pr, kFrac=kFrac, degeneracies=degeneracies) # not hermitian!


def get_momentum_bloch_realspace(lattice, cells, degeneracies, Hr, Sr, pr, kPoints):
    """calculate momentum directly from nabla_r between local orbitals"""
    Hk_at = parse_and_FT.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints, degeneracies=degeneracies)
    Sk_at = parse_and_FT.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints, degeneracies=degeneracies)
    pk_at = -1j *parse_and_FT.Rk_degenerate(cells=cells, Rr=pr, kFrac=kPoints, degeneracies=degeneracies) # not hermitian!
    vals, U = sc.linalg.eigh(Hk_at, Sk_at)
    U_dagger = np.linalg.matrix_transpose(U.conj())
    pk_bloch = np.einsum('kab,kbcz,kcd->kadz', U_dagger, pk_at, U)
    Hk_bloch = np.einsum('kab,kbc,kcd->kad', U_dagger, Hk_at, U)
    Sk_bloch = np.einsum('kab, kbc, kcd-> kad', U_dagger, Sk_at, U)
    K, a, b = Sk_bloch.shape
    I = np.eye(a, dtype=Sk_bloch.dtype)
    if not np.allclose(Sk_bloch, I):
        print("Bloch basis conversion does not leave S as identity")
    hermitian_tolerance = np.max(np.abs(pk_bloch - np.transpose(pk_bloch, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum from real space hermitian up to {hermitian_tolerance}")
    return pk_bloch
    

def to_bloch_basis(pk, Hk_orth, Sk_orth):
    vals, U = sc.linalg.eigh(Hk_orth)
    U_dagger = np.linalg.matrix_transpose(U.conj())
    pk_bloch = np.einsum('kab,kbcz,kcd->kadz', U_dagger, pk, U)
    Hk_bloch = np.einsum('kab,kbc,kcd->kad', U_dagger, Hk_orth, U)
    Sk_bloch = np.einsum('kab, kbc, kcd-> kad', U_dagger, Sk_orth, U)
    K, a, b = Sk_bloch.shape
    I = np.eye(a, dtype=Sk_bloch.dtype)
    if not np.allclose(Sk_bloch, I):
        print("Bloch basis conversion does not leave S as identity")
    return pk_bloch, Hk_bloch, Sk_bloch

def make_momentum_hermitian(pk):
    return 0.5 * (pk + np.transpose(pk, axes=(0,2,1,3)).conj())

def abs_spec_inter_intra_separate(Sr, Hr, Rr, kPoints, lattice, cells, degeneracies, range_omega, valence_idx, gamma_eV, eta_eV, T_K=0):
    """
        returns sigma tensor in cartesian coordinates 
        inter and intraband term calcualted separately
    valence_idx in python style indexing"""
    start = time.time()
    Hk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    Sk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    eigvals, U = sc.linalg.eigh(Hk_atomic, Sk_atomic)
    pk = get_momentum_bloch(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kPoints, cells=cells, lattice=lattice)
    hermitian_tolerance = np.max(np.abs(pk - np.transpose(pk, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum hermitian up to {hermitian_tolerance}")
    # pk = make_momentum_hermitian(pk=pk_bloch)
    pk = pk[:,:,:,:2]

    Nk = kPoints.shape[0]
    Nk, Nb = pk.shape[0], pk.shape[1]
    rec_lat = get_rec_lattice(lattice=lattice)
    omega = np.linspace(parse_and_FT.eV_to_au(range_omega[0]),parse_and_FT.eV_to_au(range_omega[1]),5000)
    omega = omega[omega > 1e-2]
    Nomega = omega.size
    gamma = parse_and_FT.eV_to_au(gamma_eV)
    eta = parse_and_FT.eV_to_au(eta_eV)
    omega2 = omega**2 

    bands = eigvals
    bandgap_ind = np.argmin(np.abs(bands[:,valence_idx+1]- bands[:, valence_idx]))
    Ef = (bands[bandgap_ind, valence_idx] + bands[bandgap_ind, valence_idx+1]) /2
    print(f"E_f set to {parse_and_FT.au_to_eV(Ef)}")

    sigma_tens = np.zeros((Nomega, 2, 2), dtype=complex)
    for i in range(valence_idx+1):
        f = 2 * fermi_dirac(T_K=T_K, E_state_au=bands[:,i], mu_au=Ef)
        for j in range(valence_idx+1, Nb):
            dE = bands[:, j] - bands[:, i]      # (Nk,)
            # dE_inv = np.zeros_like(dE)
            # dE_inv[dE > 1e-4] = 1/dE[dE > 1e-4]
            p = pk[:, j, i, :]             # (Nk, 3)
            # print(p)
            M_ab = np.einsum('ka,kb->abk', p, p.conj())
            denom = ((dE[None, :]**2 - omega2[:, None])**2 + gamma**2 * omega2[:, None])
            lorentz = f * gamma * omega2[:, None] / denom   # (Nomega, Nk)
            sigma_tens += np.einsum('abk,ok->oab', M_ab/dE, lorentz)
    # for i in range(Nb):
    #     dfdE = 2 * fermi_dirac_dE(T_K=T_K, E_state_au=bands[:,i], mu_au=Ef)
    #     k = np.einsum('ab,kb-> ka', rec_lat, kPoints)
    #     p = pk[:,i,i,:] - k[:,:2] # leave out factor Sk_ii because it is 1 anyway
    #     M_ab = np.einsum('ka, kb->abk', p, p.conj())
    #     lorentz = eta / ( omega2[:,None] + eta**2)
    #     sigma_tens += 0.5 * np.einsum('abk, ok -> oab', M_ab, lorentz*dfdE) 
    print(f"maximal im of sigma = {np.max(np.imag(sigma_tens))}")
    sigma_tens = np.real(sigma_tens)
    sigma_tens /= np.max(np.abs(sigma_tens))
    diagonal_mask = np.eye(sigma_tens.shape[1])
    max_offdiag = np.max(np.abs(sigma_tens - diagonal_mask * sigma_tens))
    print(f"maximal offdiagonal value of sigma: {max_offdiag}")
    end = time.time()
    print(f"Calculation took {end - start} s")
    np.save(file='sigma_tens', arr=sigma_tens)
    return parse_and_FT.au_to_eV(omega), sigma_tens

def absorption_spec_lee(Sr, Hr, Rr, kFrac, lattice, cells, degeneracies, valence_num, eta_eV, normalized=True, range_omega=(1,10), T_K=0):
    """valence_num: number of valence bands"""
    Hk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kFrac)
    Sk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kFrac)
    eigvals, U = sc.linalg.eigh(Hk_atomic, Sk_atomic)
    pk_bloch = get_momentum_bloch_lee(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
    # pk_bloch = get_momentum_bloch(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
    print(f"Max x-component {np.max(np.abs(pk_bloch[:,:,:,0]))}")
    print(f"Max y-component {np.max(np.abs(pk_bloch[:,:,:,1]))}")
    print(f"Max z-component {np.max(np.abs(pk_bloch[:,:,:,2]))}")
    
    hermitian_tolerance = np.max(np.abs(pk_bloch - np.transpose(pk_bloch, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum hermitian up to {hermitian_tolerance}")
    # pk_bloch = make_momentum_hermitian(pk=pk_bloch)

    bands = eigvals
    Nk, Nbands = np.shape(eigvals)
    omega = np.linspace(parse_and_FT.eV_to_au(range_omega[0]),parse_and_FT.eV_to_au(range_omega[1]),5000)
    nomega = np.shape(omega)[0]
    bandgap_ind = np.argmin(np.abs(bands[:,valence_num]- bands[:, valence_num-1]))
    Ef = (bands[bandgap_ind, valence_num-1] + bands[bandgap_ind, valence_num]) /2
    print(f"Fermi level at {parse_and_FT.au_to_eV(Ef)}")
    eta = parse_and_FT.eV_to_au(eta_eV)

    sigma = np.zeros(shape=(nomega, 3, 3), dtype=complex)
    for m in range(Nbands):
        fk_m = fermi_dirac(T_K=T_K, E_state_au=bands[:,m], mu_au=Ef)
        for n in range(Nbands):
            fk_n = fermi_dirac(T_K=T_K, E_state_au=bands[:,n], mu_au=Ef)
            dE = bands[:,m] - bands[:,n]
            safe_dE = np.where(np.abs(dE) > 1e-4, dE, 1.0)
            ratio = 2 * (fk_m - fk_n) / safe_dE
            fermi_fac = np.where(np.abs(dE) > 1e-4,
                                 ratio,
                                 2*fermi_dirac_dE(T_K=T_K, E_state_au=bands[:,n], mu_au=Ef))
            pk1 = pk_bloch[:,m,n,:]
            pk2 = pk_bloch[:,n,m,:]
            pk_prod = np.einsum('ka, kb -> kab', pk1, pk2)
            dE = bands[:,m] - bands[:,n]
            denominator = 1 /(omega[:,None] + dE[None,:] + 1j*eta)
            sigma += -1j * np.einsum('kab,k,ok ->oab ', pk_prod, fermi_fac, denominator)
    sigma = np.real(sigma)
    if normalized:
        sigma /= np.max(np.abs(sigma[:,:2,:2]))
    return parse_and_FT.au_to_eV(omega), sigma

def absorption_spec_momentum(Sr, Hr, pr, kFrac, lattice, cells, degeneracies, valence_num, eta_eV, range_omega=(1,10), T_K=0):
    """
    absorption spectrum with real-space momentum elements directly from file
    valence_num: number of valence bands"""
    Hk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kFrac)
    Sk_atomic = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kFrac)
    eigvals, U = sc.linalg.eigh(Hk_atomic, Sk_atomic)
    pk_bloch = get_momentum_bloch_realspace(Hr=Hr, Sr=Sr, pr=pr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)
    print(f"Max x-component {np.max(np.abs(pk_bloch[:,:,:,0]))}")
    print(f"Max y-component {np.max(np.abs(pk_bloch[:,:,:,1]))}")
    print(f"Max z-component {np.max(np.abs(pk_bloch[:,:,:,2]))}")
    
    hermitian_tolerance = np.max(np.abs(pk_bloch - np.transpose(pk_bloch, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum hermitian up to {hermitian_tolerance}")
    # pk_bloch = make_momentum_hermitian(pk=pk_bloch)

    bands = eigvals
    Nk, Nbands = np.shape(eigvals)
    omega = np.linspace(parse_and_FT.eV_to_au(range_omega[0]),parse_and_FT.eV_to_au(range_omega[1]),5000)
    nomega = np.shape(omega)[0]
    bandgap_ind = np.argmin(np.abs(bands[:,valence_num]- bands[:, valence_num-1]))
    Ef = (bands[bandgap_ind, valence_num-1] + bands[bandgap_ind, valence_num]) /2
    print(f"Fermi level at {parse_and_FT.au_to_eV(Ef)}")
    eta = parse_and_FT.eV_to_au(eta_eV)

    sigma = np.zeros(shape=(nomega, 3, 3), dtype=complex)
    for m in range(Nbands):
        fk_m = fermi_dirac(T_K=T_K, E_state_au=bands[:,m], mu_au=Ef)
        for n in range(Nbands):
            fk_n = fermi_dirac(T_K=T_K, E_state_au=bands[:,n], mu_au=Ef)
            dE = bands[:,m] - bands[:,n]
            safe_dE = np.where(np.abs(dE) > 1e-4, dE, 1.0)
            ratio = 2 * (fk_m - fk_n) / safe_dE
            fermi_fac = np.where(np.abs(dE) > 1e-4,
                                 ratio,
                                 2*fermi_dirac_dE(T_K=T_K, E_state_au=bands[:,n], mu_au=Ef))
            pk1 = pk_bloch[:,m,n,:]
            pk2 = pk_bloch[:,n,m,:]
            pk_prod = np.einsum('ka, kb -> kab', pk1, pk2)
            dE = bands[:,m] - bands[:,n]
            denominator = 1 /(omega[:,None] + dE[None,:] + 1j*eta)
            sigma += -1j * np.einsum('kab,k,ok ->oab ', pk_prod, fermi_fac, denominator)
    sigma = np.real(sigma)
    sigma /= np.max(np.abs(sigma[:,:2,:2]))
    return parse_and_FT.au_to_eV(omega), sigma

def spectrum_in_direction(sigma_tens, vec):
    """calculate the spectrum in a certain direction. Assumes that the tensors 
    components are with respect to cartesian basis vectors"""
    unit_vec = vec/np.linalg.norm(vec)
    return np.einsum('a, oab, b-> o', unit_vec, sigma_tens, unit_vec)