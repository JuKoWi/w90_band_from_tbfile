import w90
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import time
import matplotlib.colors as mcolors

MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
             'M' : np.array([0.5, 0, 0]),
             'K' : np.array([1/3, 1/3, 0]),
            }

def parse_dftb_band(filepath, n_bands):
    with open(file=filepath, mode='r') as f:
        bandstructure = []
        while True:
            comment = f.readline()
            if not comment:
                break
            if "KPT" in comment:
                k_point_levels = []
                for _ in range(n_bands):
                    line = f.readline()
                    number = float(line.split()[1])
                    k_point_levels.append(number)
                bandstructure.append(k_point_levels)
    return np.array(bandstructure)

def plotLines(ax, pos, labels):
    for p in pos:
        ax.axvline(x=p, color='k')
    ax.set_xticks(pos, labels)
    ax.set_xlim([pos[0], pos[-1]])
    ax.set_ylabel("E [eV]", labelpad=-5)

def parsePath(path, lattice, labelToK, pointsPerSegment=100):
    """use this after w90.py"""
    recipLattice = 2*np.pi * np.linalg.inv(lattice).T
    segments = []
    lastRelPos = 0
    labels = [ [path[0], lastRelPos] ]
    for s, e in zip(path[:-1], path[1:]):
        if not s.isalpha() or not e.isalpha():
            continue
        if labels[-1][0] != s:
            labels[-1][0] += "|" + s
        start = labelToK[s]
        end = labelToK[e]
        h = np.linspace(0, 1, pointsPerSegment)
        relPos = lastRelPos + h * np.linalg.norm(recipLattice.T @ (end-start))
        kPoints = start + h[:, None] * (end-start)
        segments.append( [kPoints, relPos] )
        # print(f"{s} -> {e} : {kPoints[0]} -> {kPoints[-1]}")
        lastRelPos = relPos[-1]
        labels.append([e, lastRelPos])
    # normalize pos
    for s in segments:
        s[1] /= lastRelPos
    for l in labels:
        l[1] /= lastRelPos
    return segments, labels

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
    Hk = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints, degeneracies=degeneracies)
    Sk = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints, degeneracies=degeneracies)
    Rk = w90.Rk_degenerate(cells=cells, Rr=Rr, kFrac=kPoints, degeneracies=degeneracies) # not hermitian!
    A1 = np.all(sc.linalg.ishermitian(Sk, atol=1e-12))
    B1 = np.all(sc.linalg.ishermitian(Hk, atol=1e-12))
    if not (A1 and B1):
        print("H and S not hermitian")
    S_inv_sqrt = diagonalization_inv_sqrt(Sk)
    if not np.all(sc.linalg.ishermitian(S_inv_sqrt, atol=1e-12)):
        print("Orthogonalization matrix not hermitian")
    grad_S_inv_sqrt, grad_H = grad_Hk_S_invsqrt(Hr=Hr, Sr=Sr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    assert np.allclose(grad_H, alternative_grad_Hk(Hr=Hr, Sr=Sr, kPoints=kPoints, lattice=lattice, cells=cells, degeneracies=degeneracies))
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
    H_plus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_plus, degeneracies=degeneracies) 
    H_minus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_minus, degeneracies=degeneracies)
    H_2plus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    H_2minus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    derivative = (8 * H_plus - 8 * H_minus + H_2minus - H_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    if not np.all(sc.linalg.ishermitian(derivative, atol=1e-12)):
        print("partial Hk not hermitian")
    return derivative

def alternative_grad_Hk(Hr, Sr, lattice, kPoints, cells, degeneracies):
    """Alternative way to calculate grad H in orthogonal basis"""
    Sk = w90.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    Hk = w90.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    S_inv_sqrt = diagonalization_inv_sqrt(Sk=Sk)
    S_inv_sqrt_grad = gradient_S_inv_sqrt(Sr, kPoints=kPoints, lattice=lattice, cells=cells, degeneracies=degeneracies)
    gradHk = gradient_H_atomic(Hr=Hr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    analytical_gradHk = w90.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr,kFrac=kPoints, lattice=lattice)
    assert np.allclose(analytical_gradHk, gradHk)
    A = np.einsum('kabx, kbc, kcd -> kadx', S_inv_sqrt_grad, Hk, S_inv_sqrt)
    B = np.einsum('kab,kbcx,kcd->kadx', S_inv_sqrt, gradHk, S_inv_sqrt)
    C = np.einsum('kab,kbc,kcdx->kadx', S_inv_sqrt, Hk, S_inv_sqrt_grad)
    return A+B+C

def partial_Hk_S_invsqrt(Hr, Sr, kPoints, cells, component, degeneracies):
    stencil_shift_factor = 1e-3 # smaller stencil shift does not make sense
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    H_plus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_plus, degeneracies=degeneracies) 
    H_minus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_minus, degeneracies=degeneracies)
    H_2plus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    H_2minus = w90.Hk_degenerate(cells=cells, Hr=Hr, kFrac=kPoints_2minus, degeneracies=degeneracies)
    S_plus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_plus, degeneracies=degeneracies)
    S_minus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_minus, degeneracies=degeneracies)
    S_2plus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    S_2minus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2minus, degeneracies=degeneracies)
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

def partial_Sminushalf(Sr, kPoints, cells, degeneracies, component):
    stencil_shift_factor = 1e-3 #smaller stencil shift does not make sense
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    S_plus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_plus, degeneracies=degeneracies)
    S_minus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_minus, degeneracies=degeneracies)
    S_2plus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2plus, degeneracies=degeneracies)
    S_2minus = w90.Hk_degenerate(cells=cells, Hr=Sr, kFrac=kPoints_2minus, degeneracies=degeneracies)
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

def diagonalization_inv_sqrt(Sk):
    vals, vecs = sc.linalg.eigh(Sk)
    U = vecs
    Udagger = np.linalg.matrix_transpose(vecs.conj())
    minus_half = 1/np.sqrt(vals) 
    N,M = vals.shape
    diags_minushalf = np.zeros((N,M,M), dtype=complex)
    idx = np.arange(M)
    diags_minushalf[:,idx, idx] = minus_half
    S_minushalf = np.einsum('kab, kbc, kcd -> kad', U, diags_minushalf, Udagger)
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

def get_dipole_orth(Hr, Sr, Rr, kPoints, cells, lattice, degeneracies):
    Sk_orth, Hk_orth, dk_orth, grad_H = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints, degeneracies=degeneracies)
    return dk_orth, Sk_orth, Hk_orth

def get_dipole_atomic(Rr, Sr, kPoints, cells, lattice, degeneracies):
    gradS = w90.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints, lattice=lattice)
    gradS_alt = gradient_H_atomic(Hr=Sr, kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    assert np.allclose(gradS, gradS_alt)
    Rk = w90.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kPoints)
    return Rk + 1j* gradS

def get_momentum(Hr, Sr, Rr, degeneracies, kPoints, cells, lattice):
    Sk_orth, Hk_orth, dk_orth, grad_H = orthogonalize(lattice=lattice, cells=cells, degeneracies=degeneracies, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints)
    if not check_vector_hermitian(pk=grad_H, atol=1e-10):
        print("gradient H not hermitian")
    commutator = np.einsum('kabz, kbc->kacz', dk_orth, Hk_orth) - np.einsum('kab, kbcz -> kacz', Hk_orth, dk_orth)
    if not check_vector_hermitian(pk=1j*commutator, atol=1e-8):
        print("i * [d, H] is not hermitian")
    p = +1j * commutator + grad_H
    return p, Sk_orth, Hk_orth

def alternative_momentum(Hr, Sr, Rr, degeneracies, kPoints, cells, lattice):
    Sk = w90.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kPoints)
    Rk = w90.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kPoints)
    Hk = w90.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints)
    gradHk = w90.grad_H_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kPoints, lattice=lattice)
    eigvals, U = sc.linalg.eigh(Hk, Sk)
    U_dagger = np.transpose(U, axes=(0,2,1))
    term1 = np.einsum("kab, kbcz, kcd -> kadz", U, gradHk, U_dagger)
    term2 = 1j * np.einsum("kab, kbcz, kcd, ka -> kadz", U, np.transpose(Rk, axes=(0,2,1,3)).conj(), U_dagger, eigvals)
    term3 = -1j * np.einsum('kab, kbcz, kcd, kd -> kadz', U, Rk, U_dagger, eigvals)
    assert check_vector_hermitian(pk = term1 + term2 + term3, atol=1e-7)
    return term1 + term2 + term3

def get_rec_lattice(lattice):
    """return matrix with rec. lattice vectors as columns
        ATTENTION: Takes lattice as matrix with lattice vectors as ROWS(!)
        and considers factor of 2 pi
    """
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    assert np.allclose(np.linalg.inv(lattice.T) * 2 * np.pi, [b1, b2, b3])
    return np.array([b1, b2, b3]).T

def to_bloch_basis(pk, Hk_orth, Sk_orth):
    vals, U = sc.linalg.eigh(Hk_orth)
    U_dagger = np.linalg.matrix_transpose(U.conj())
    pk_bloch = np.einsum('kab,kbcz,kcd->kadz', U_dagger, pk, U)
    Hk_bloch = np.einsum('kab,kbc,kcd->kad', U_dagger, Hk_orth, U)
    Sk_bloch = np.einsum('kab, kbc, kcd-> kad', U_dagger, Sk_orth, U)
    K, a, b = Sk_bloch.shape
    I = np.eye(a, dtype=Sk_bloch.dtype)
    assert np.allclose(Sk_bloch, I)
    return pk_bloch, Hk_bloch, Sk_bloch

def k_grid(n_points):
    """returns list of k-points, x-component changes fastest from low to high
    due to periodicity, BZ can be shifted so it starts at gamma point"""
    x = np.arange(n_points[0])/n_points[0]
    y = np.arange(n_points[1])/n_points[1]
    z = np.arange(n_points[2])/n_points[2]
    xv, yv, zv = np.meshgrid(x,y,z) 
    xv = xv.flatten(order='C')
    yv = yv.flatten(order='C')
    zv = zv.flatten(order='C')
    kpoints = np.column_stack((xv,yv,zv))
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(kpoints[:,0], kpoints[:,1], kpoints[:,2])
    # plt.show()
    return kpoints

#TODO: function to calculate chemical potential

def fermi_dirac(T_K, E_state_au, mu_au):
    mu_eV = w90.au_to_eV(mu_au)
    E_state_eV = w90.au_to_eV(E_state_au)
    kb = sc.constants.physical_constants['Boltzmann constant in eV/K'][0]
    if T_K == 0:
        f = np.zeros_like(E_state_eV, dtype=float)
        f[E_state_eV < mu_eV] = 1.0
        f[E_state_eV == mu_eV] = 0.5
        return f
    f = 1/(np.exp((E_state_eV - mu_eV )/ (T_K * kb))+1)
    return f

def fermi_dirac_dE(T_K, E_state_au, mu_au):
    E_state_eV = w90.au_to_eV(E_state_au)
    mu_eV = w90.au_to_eV(mu_au)
    kb = sc.constants.physical_constants['Boltzmann constant in eV/K'][0]
    if T_K == 0:
        return np.zeros_like(E_state_eV, dtype=float)
    x = (E_state_eV - mu_eV) / (kb * T_K)
    f_eV = -1.0 / (4 * kb * T_K * np.cosh(x / 2)**2)  
    f_au = f_eV * w90.au_to_eV(1.0)
    return f_au

def check_vector_hermitian(pk, atol):
    is_hermitian = True
    for i in range(3):
        p = pk[:,:,:,i]
        if not np.all(sc.linalg.ishermitian(p, atol=atol)):
            is_hermitian = False
    return is_hermitian

def make_momentum_hermitian(pk):
    return 0.5 * (pk + np.transpose(pk, axes=(0,2,1,3)).conj())

def absorption_spec_simple(Sr, Hr, Rr, kPoints, lattice, cells, degeneracies, range_omega, valence_idx, gamma_eV):
    start = time.time()
    pk, Sk_orth, Hk_orth = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth, Sk_orth=Sk_orth)
    hermitian_tolerance = np.max(np.abs(pk_bloch - np.transpose(pk_bloch, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum hermitian up to {hermitian_tolerance}")
    # pk = make_momentum_hermitian(pk_bloch)
    pk = pk[:,:,:,:2]

    Nk = kPoints.shape[0]
    Nk, Nb = pk_bloch.shape[0], pk_bloch.shape[1]
    rec_lat = get_rec_lattice(lattice=lattice)
    omega = np.linspace(w90.eV_to_au(range_omega[0]),w90.eV_to_au(range_omega[1]),5000)
    omega = omega[omega > 1e-2]
    Nomega = omega.size
    gamma = w90.eV_to_au(gamma_eV)
    omega2 = omega**2 

    bands = np.real(np.einsum('kaa->ka', Hk_bloch))
    sigma_tens = np.zeros((Nomega, 3, 3), dtype=complex)
    """
        for 200x200 k and 5000 omega: 203 s 
    """
    print(valence_idx)
    for i in range(valence_idx + 1):
        for j in range(valence_idx+1, Nb):
            print(f"{i} {j}")
            dE = bands[:, j] - bands[:, i]      # (Nk,)
            dE_inv = 1/dE
            dE_inv[dE_inv > 1e4] = 0
            p = pk_bloch[:, j, i, :]             # (Nk, 3)
            M_ab = np.einsum('ka,kb->abk', p, p.conj(), optimize=True)
            denom = ((dE[None, :]**2 - omega2[:, None])**2
                     + gamma**2 * omega2[:, None])
            lorentz = gamma * omega2[:, None] / denom   # (Nomega, Nk)
            sigma_tens += np.einsum('abk,ok->oab', M_ab * dE_inv, lorentz)
    sigma_tens = np.real(sigma_tens) 
    sigma_tens /= np.max(np.abs(sigma_tens))
    end = time.time()
    print(f"Calculation took {end - start} s")
    return w90.au_to_eV(omega), sigma_tens

def absorption_spec(Sr, Hr, Rr, kPoints, lattice, cells, degeneracies, range_omega, valence_idx, gamma_eV, eta_eV, T_K=0):
    """
        returns sigma tensor in cartesian coordinates 
    valence_idx in python style indexing"""
    start = time.time()
    pk_orth, Sk_orth, Hk_orth = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr,kPoints=kPoints, cells=cells, lattice=lattice, degeneracies=degeneracies)
    pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk_orth, Hk_orth=Hk_orth, Sk_orth=Sk_orth)
    alt_momentum = alternative_momentum(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kPoints, cells=cells, lattice=lattice)
    print(np.allclose(pk_bloch, alt_momentum))
    hermitian_tolerance = np.max(np.abs(pk_bloch - np.transpose(pk_bloch, axes=(0,2,1,3)).conj()))
    print(f"Bloch-basis momentum hermitian up to {hermitian_tolerance}")
    pk = make_momentum_hermitian(alt_momentum)
    # pk = pk[:,:,:,:2]

    Nk = kPoints.shape[0]
    Nk, Nb = pk_bloch.shape[0], pk_bloch.shape[1]
    rec_lat = get_rec_lattice(lattice=lattice)
    print(rec_lat)
    omega = np.linspace(w90.eV_to_au(range_omega[0]),w90.eV_to_au(range_omega[1]),5000)
    omega = omega[omega > 1e-2]
    Nomega = omega.size
    gamma = w90.eV_to_au(gamma_eV)
    eta = w90.eV_to_au(eta_eV)
    omega2 = omega**2 

    bands = np.real(np.einsum('kaa->ka', Hk_bloch))
    bandgap_ind = np.argmin(np.abs(bands[:,valence_idx+1]- bands[:, valence_idx]))
    Ef = (bands[bandgap_ind, valence_idx] + bands[bandgap_ind, valence_idx+1]) /2
    print(f"E_f set to {w90.au_to_eV(Ef)}")

    sigma_tens = np.zeros((Nomega, 3, 3), dtype=complex)
    for i in range(valence_idx+1):
        f = 2 * fermi_dirac(T_K=T_K, E_state_au=bands[:,i], mu_au=Ef)
        for j in range(valence_idx+1, Nb):
            dE = bands[:, j] - bands[:, i]      # (Nk,)
            # dE_inv = np.zeros_like(dE)
            # dE_inv[dE > 1e-4] = 1/dE[dE > 1e-4]
            p = pk[:, i, j, :]             # (Nk, 3)
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
    print(f"maximal im of eps2 = {np.max(np.imag(sigma_tens))}")
    sigma_tens = np.real(sigma_tens)
    sigma_tens /= np.max(np.abs(sigma_tens))
    diagonal_mask = np.eye(sigma_tens.shape[1])
    max_offdiag = np.max(np.abs(sigma_tens - diagonal_mask * sigma_tens))
    print(f"maximal offdiagonal value of sigma: {max_offdiag}")
    end = time.time()
    print(f"Calculation took {end - start} s")
    np.save(file='sigma_tens', arr=sigma_tens)
    return w90.au_to_eV(omega), sigma_tens

def plot_bands(segments, labels, bandstructures:list, pltname):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scale = 800
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for j, bands in enumerate(bandstructures):
        for i, (kPoints, relPos) in enumerate(segments):
            ax.plot(relPos, bands[i], '.', color=colors[names[j]],ms=1)
    fig.tight_layout()
    l, pos = zip(*labels)
    point_symbols = []
    for i in labels:
        point_symbols.append(i[0])
    plotLines(ax=ax, pos=pos, labels=point_symbols)
    plt.savefig(f'{pltname}.pdf')
    plt.show()

def spectrum_in_direction(sigma_tens, vec):
    """calculate the spectrum in a certain direction. Assumes that the tensors 
    components are with respect to cartesian basis vectors"""
    unit_vec = vec/np.linalg.norm(vec)
    return np.einsum('a, oab, b-> o', unit_vec, sigma_tens, unit_vec)


if __name__ == "__main__":
    # segments, labels, bands_orth, H_orth, S_orth, d_orth =  bandstructure_orth_basis(lattice, cells, Hr, Sr, Rr)
    # bands_alex = parse_dftb_band(filepath="band_mos2_alex_27band.out", n_bands=27) 
    # bands_alex = [bands_alex[:100], bands_alex[100:200], bands_alex[200:300]]
    # bands_own = parse_dftb_band(filepath="band_mos2_own_27band_denssup_corrected_eigval.out", n_bands=27)
    # bands_own = [bands_own[:100], bands_own[100:200], bands_own[200:300]]


    lattice, cells, degeneracies2, Hr, Sr, Rr = w90.read_tb("seedname_mos2_ase.dat")
    # dk1, Sk_orth, Hk_orth= get_dipole(degeneracies=degeneracies2, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=[[0.2, 0, 0]], cells=cells, lattice=lattice)
    # dk2, Sk_orth, Hk_orth= get_dipole(degeneracies=degeneracies2, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=[[-0.2, 0, 0]], cells=cells, lattice=lattice)
    # dkA = dk1[0,1,0]
    # dkB = dk2[0,1,0]
    # print(dkA)
    # print(dkB)
    # sys.exit()
    kcart1 = np.array([0.5,0,0])
    kcart2 = np.array([-0.5,0,0])
    kfrac1 = 1/(2*np.pi) * lattice @ kcart1
    kfrac2 = 1/(2*np.pi) * lattice @ kcart2
    omega, sigma_tens2 = absorption_spec(Sr=Sr, 
                                        Hr=Hr, 
                                        Rr=Rr, 
                                        kPoints=k_grid(n_points=(10, 10, 1)), 
                                        # kPoints=np.array([kfrac1, kfrac2 ]),
                                        lattice=lattice,
                                        cells=cells,
                                        range_omega=(0, 10),
                                        valence_idx=8,
                                        gamma_eV=0.1, 
                                        eta_eV=0.1, 
                                        degeneracies=degeneracies2)

    # rot120deg = np.array([[-0.5, -np.sqrt(3)/2], [np.sqrt(3)/2, -0.5]])
    # sigma_tens2 = np.einsum('ab, obc, cd -> oad', rot120deg.T, sigma_tens2, rot120deg )
    print(sigma_tens2[2000])
    # print(sigma_tens2[2000])
    # plt.plot(omega, spectrum_in_direction(sigma_tens=sigma_tens2, vec=[1,0]))
    # plt.plot(omega, spectrum_in_direction(sigma_tens=sigma_tens2, vec=[-0.5,np.sqrt(3)/2]))
    # plt.show()

    # plt.plot(omega, sigma_tens[:,0,1], 
    #          '.',
    #            ms=1)
    plt.plot(omega, sigma_tens2[:,1,0],
             '.',
               ms=1)
    plt.plot(omega, sigma_tens2[:,0,1],
             '.',
               ms=1)
    plt.show()


    # plt.plot(omega, sigma_tens[:,1,0], 
    #          '.',
    #            ms=1)
    plt.plot(omega, sigma_tens2[:,0,0],
             '.',
               ms=1)
    plt.plot(omega, sigma_tens2[:,1,1],
             '.',
               ms=1)
    plt.show()