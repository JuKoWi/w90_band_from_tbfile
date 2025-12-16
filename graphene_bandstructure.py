import w90
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
# from bandstructure import parsePath,plotLines

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

def orthogonalize_basis(lattice, cells, Hr, Sr, Rr):
    segments, labels = parsePath("GMKG", lattice=lattice, labelToK=MoS2_labelToK)
    bands_orth = []
    R_orth = []
    H_orth = []
    S_orth = []
    for i, (kPoints, relPos) in enumerate(segments):
        Sk_orth, Hk_orth, Rk_orth = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints)
        vals, vecs = sc.linalg.eigh(Hk_orth)
        bands_orth.append(np.real(vals))
        H_orth.append(Hk_orth)
        S_orth.append(Sk_orth)
        R_orth.append(Rk_orth)
    return segments, labels, bands_orth, H_orth, S_orth, R_orth 

def orthogonalize(lattice, cells, Hr, Sr, Rr, kPoints):
    Hk = w90.Hk(cells=cells, H=Hr, kFrac=kPoints)
    Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
    Rk = w90.Dk(cells=cells, D=Rr, kFrac=kPoints)
    S_inv_sqrt = diagonalization_inv_sqrt(Sk)
    S_inv_sqrt_grad = gradient_S_inv_sqrt(Sr, kPoints=kPoints, lattice=lattice)
    Hk_orth = S_inv_sqrt @ Hk @ S_inv_sqrt
    Sk_orth = S_inv_sqrt @ Sk @ S_inv_sqrt
    S_minushalf_dagger = S_inv_sqrt
    Rk_orth = np.einsum('kab,kbc,kcdz->kadz', S_minushalf_dagger, Sk, S_inv_sqrt_grad) + np.einsum('kab,kbcz,kcd->kadz',S_minushalf_dagger, Rk, S_inv_sqrt)
    test_diagonal = np.tile(np.eye(np.shape(Sk_orth)[1]), (np.shape(Sk_orth)[0], 1, 1))
    return Sk_orth, Hk_orth, Rk_orth

def get_momentum(Hr, Sr, Rr, kPoints, cells, lattice):
    Sk_orth, Hk_orth, Rk_orth = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints)
    grad = gradient_H(Hr=Hr, kPoints=kPoints, lattice=lattice)
    print(np.shape(grad))
    print(np.shape(Rk_orth))
    commutator = np.einsum('kabz, kbc->kacz', Rk_orth, Hk_orth) - np.einsum('kab, kbcz -> kacz', Hk_orth, Rk_orth)
    p = 1j*commutator - grad
    return p

def partial_Hk(Hr, kPoints, component):
    stencil_shift_factor = 0.001
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    H_plus = w90.Hk(cells=cells, H=Hr, kFrac=kPoints_plus)
    H_minus = w90.Hk(cells=cells, H=Hr, kFrac=kPoints_minus)
    H_2plus = w90.Hk(cells=cells, H=Hr, kFrac=kPoints_2plus)
    H_2minus = w90.Hk(cells=cells, H=Hr, kFrac=kPoints_2minus)
    derivative = (8 * H_plus - 8 * H_minus + H_2minus - H_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    return derivative

def partial_Sminushalf(Sr, kPoints, component):
    stencil_shift_factor = 0.001
    basis_vec = np.zeros((3,))
    basis_vec[component] = 1
    stencil_shift = stencil_shift_factor * basis_vec
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    S_plus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_plus)
    S_minus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_minus)
    S_2plus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_2plus)
    S_2minus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_2minus)
    Shalf_plus = diagonalization_inv_sqrt(S_plus)
    Shalf_minus = diagonalization_inv_sqrt(S_minus)
    Shalf_2plus = diagonalization_inv_sqrt(S_2plus)
    Shalf_2minus = diagonalization_inv_sqrt(S_2minus)
    derivative = (8 * Shalf_plus - 8 * Shalf_minus + Shalf_2minus - Shalf_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    return derivative

def gradient_H(Hr, kPoints, lattice):
    """Returns array of shape (kpoints, basis, basis, components)"""
    rec_lattice = get_rec_lattice(lattice=lattice)
    canonical_to_rec_lattice = rec_lattice.T
    Ha = partial_Hk(Hr=Hr, kPoints=kPoints, component=0)
    Hb = partial_Hk(Hr=Hr, kPoints=kPoints, component=1)
    Hc = partial_Hk(Hr=Hr, kPoints=kPoints, component=2)
    grad_S_inv_sqrt = np.stack((Ha, Hb, Hc), axis=-1)
    grad_S_inv_sqrt = np.einsum('kabc, cd ->kabd', grad_S_inv_sqrt, canonical_to_rec_lattice)
    return grad_S_inv_sqrt

def gradient_S_inv_sqrt(Sr, kPoints, lattice):
    """Returns array of shape (kpoints, basis, basis, components)"""
    rec_lattice = get_rec_lattice(lattice=lattice)
    canonical_to_rec_lattice = rec_lattice.T

    Sa = partial_Sminushalf(Sr=Sr, kPoints=kPoints, component=0)
    Sb = partial_Sminushalf(Sr=Sr, kPoints=kPoints, component=1)
    Sc = partial_Sminushalf(Sr=Sr, kPoints=kPoints, component=2)
    grad_S_inv_sqrt = np.stack((Sa, Sb, Sc), axis=-1)
    grad_S_inv_sqrt = np.einsum('kabc, cd ->kabd', grad_S_inv_sqrt, canonical_to_rec_lattice)
    return grad_S_inv_sqrt

def get_rec_lattice(lattice):
    a1, a2, a3 = lattice
    volume = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    return np.array([b1, b2, b3])

def diagonalization_inv_sqrt(Sk):
    vals, vecs = sc.linalg.eig(Sk)
    U = vecs
    Udagger = np.linalg.matrix_transpose(vecs.conj())
    minus_half = 1/np.sqrt(vals) 
    N,M = vals.shape
    diags_minushalf = np.zeros((N,M,M), dtype=complex)
    idx = np.arange(M)
    diags_minushalf[:,idx, idx] = minus_half
    S_minushalf = np.einsum('kab, kbc, kcd -> kad', U, diags_minushalf, Udagger)
    return S_minushalf

def to_bloch_basis(pk, Hk_orth):
    vals, U = sc.linalg.eigh(Hk_orth)
    U_dagger = np.linalg.matrix_transpose(U.conj())
    pk_bloch = np.einsum('kab,kbcz,kcd->kadz', U_dagger, pk, U)
    Hk_bloch = np.einsum('kab,kbc,kcd->kad', U_dagger, Hk_orth, U)
    return pk_bloch, Hk_bloch 

def get_absorption_spectrum(Sr, Hr, Rr, kPoints, lattice, cells, range_omega, valence_idx):
    pk = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints, cells=cells, lattice=lattice)
    Sk_orth, Hk_orth, Rk_orth = orthogonalize(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kPoints)
    num_k = np.shape(kPoints)[0]
    pk_bloch, Hk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth) 
    N_bands = np.shape(Sk_orth)[1]
    bands = np.real(np.einsum('kaa->ka', Hk_bloch))
    upper_omega = w90.eV_to_au(range_omega[1])
    lower_omega = w90.eV_to_au(range_omega[0])
    omega = np.linspace(start=lower_omega, stop=upper_omega, num=1000)
    eps_tens = np.zeros((3,3,np.shape(omega)[0]), dtype=complex)
    cond_bands = N_bands - (valence_idx + 1)
    N_excite = cond_bands * (valence_idx + 1)
    M_alpha = np.zeros((num_k, 3, N_excite), dtype=complex)
    M_beta = np.zeros((num_k, 3, N_excite), dtype=complex)
    delta_E = np.zeros((num_k, N_excite))
    count_excite = 0
    for i in range(valence_idx + 1):
        for j in range(valence_idx+1, N_bands):
            M_alpha[:,:, count_excite] = pk_bloch[:,j,i,:]
            M_beta[:,:, count_excite] = pk_bloch[:,j,i,:].conj()
            delta_E[:,count_excite] = bands[:,j] - bands[:,i]
            count_excite += 1
    assert count_excite == N_excite
    M_alpha_beta = np.einsum('kab,kcb->kacb', M_alpha, M_beta)
    gamma = 0.0001
    omega_reshaped = omega[:,np.newaxis, np.newaxis]
    lorentz_term = gamma * omega_reshaped /(((delta_E[np.newaxis,:,:])**2 - omega_reshaped**2)**2 + gamma * omega_reshaped**2)
    eps_tens = np.einsum('kabc,okc->oab', M_alpha_beta, lorentz_term)
    omega_eV = w90.au_to_eV(omega)
    return omega_eV, np.real(eps_tens)


def k_grid(n_points):
    x = np.linspace(-0.5, 0.5, num=n_points[0])
    y = np.linspace(-0.5, 0.5, num=n_points[1])
    z = np.linspace(-0.5, 0.5, num=n_points[2])
    xv, yv, zv = np.meshgrid(x,y,z) 
    xv = xv.flatten()
    yv = yv.flatten()
    zv = zv.flatten()
    kpoints = np.column_stack((xv,yv,zv))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(kpoints[:,0], kpoints[:,1], kpoints[:,2])
    plt.show()
    return kpoints

if __name__ == "__main__":
    lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb("seedname_tb.dat")
    MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
                 'M' : np.array([0.5, 0, 0]),
                 'K' : np.array([1/3, 1/3, 0]),
                }
    segments, labels, bands_orth, H_orth, S_orth, R_orth =  orthogonalize_basis(lattice, cells, Hr, Sr, Rr)
    omega, eps_tens = get_absorption_spectrum(Sr=Sr, Hr=Hr, Rr=Rr, kPoints=k_grid(n_points=(40,40,1)), lattice=lattice, cells=cells, range_omega=(0, 30), valence_idx=3)
    plt.plot(omega, eps_tens[:,0,0], '.', ms=1)
    plt.plot(omega, eps_tens[:,1,1], '.', ms=1)
    plt.plot(omega, eps_tens[:,2,2], '.', ms=1)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scale = 800
    for i, (kPoints, relPos) in enumerate(segments):
        # Hk = w90.Hk(cells=cells,H=Hr, kFrac=kPoints)
        # Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
        # vals, vecs = sp.linalg.eig(Hk, Sk)
        # ax.plot(relPos, vals_dftb[i], '.', color="blue", ms=1)
        # ax.plot(relPos, vals, '.', color='orange', ms=1)
        ax.plot(relPos, bands_orth[i], '.', color='blue',ms=1)
        # ax.plot(relPos, bands2[i], '.', color='orange', ms=1)
        # ax.plot(relPos, S_eigvals[i], '.', color='orange', ms=1)
        Rk = R_orth[i]
        # print(np.shape(Rk[:,0,0,0]))
        # ax.plot(relPos, Rk[:,0,0,0], '.', color='orange', ms=1)
        # ax.plot(relPos, Rk[:,0,0,1], '.', color='blue',  ms=1)
        # ax.plot(relPos, Rk[:,0,0,2], '.', color='green', ms=1)
    fig.tight_layout()
    l, pos = zip(*labels)
    point_symbols = []
    for i in labels:
        point_symbols.append(i[0])
    plotLines(ax=ax, pos=pos, labels=point_symbols)
    # plt.show()
