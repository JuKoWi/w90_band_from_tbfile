import w90
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from bandstructure import parsePath,plotLines

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

def orthogonalize_basis(lattice, cells, degeneracy, Hr, Sr, Rr):
    segments, labels = parsePath("GMKG", lattice=lattice, labelToK=MoS2_labelToK)
    bands_orth = []
    bands_direct = []
    dipole_moment = []
    S_eigvals = []
    for i, (kPoints, relPos) in enumerate(segments):
        Hk = w90.Hk(cells=cells,H=Hr, kFrac=kPoints)
        Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
        vals_S,vecs = sc.linalg.eigh(Sk)
        # for ki in range(len(kPoints)):
        #     print(sc.linalg.ishermitian(Hk[ki], atol=1e-10))
        #     print(sc.linalg.ishermitian(Sk[ki], atol=1e-10))
        Rk = w90.Dk(cells=cells, D=Rr, kFrac=kPoints)
        print(Rk.shape)
        S_inv_sqrt = diagonalization_inv_sqrt(Sk)
        S_inv_sqrt_grad = gradient_S_inv_sqrt(Sr, kPoints=kPoints, lattice=lattice)
        Hk_orth = S_inv_sqrt @ Hk @ S_inv_sqrt
        S_minushalf_dagger = np.transpose(S_inv_sqrt.conj(), axes=(0,2,1)) # useless since hermitian?
        Rk_orth = np.einsum('kab,kbc,kcdz->kadz', S_minushalf_dagger, Sk, S_inv_sqrt_grad) + np.einsum('kab,kbcz,kcd->kadz',S_minushalf_dagger, Rk, S_inv_sqrt)
        # for ki in range(np.shape(Hk_orth)[0]):
        #     print(sc.linalg.ishermitian(S_inv_sqrt[ki], atol=1e-10))
            # print(sc.linalg.ishermitian(Hk_orth[ki], atol=1e-10))
            # print(np.allclose(Sk[ki] @ S_inv_sqrt[ki] @ S_inv_sqrt[ki], np.eye(8), atol=1e-10))
        vals, vecs = sc.linalg.eigh(Hk_orth)
        vals1, vecs1 = sc.linalg.eig(Hk, Sk)
        bands_orth.append(np.real(vals))
        bands_direct.append(np.real(vals1))
        dipole_moment.append(Rk_orth)
        S_eigvals.append(vals_S)

    return segments, labels, bands_orth, bands_direct, dipole_moment, S_eigvals


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

def gradient_S_inv_sqrt(Sr, kPoints, lattice):
    """Returns array of shape (kpoints, basis, basis, components)"""
    rec_lattice = get_rec_lattice(lattice=lattice)
    canonical_to_rec_lattice = rec_lattice.T
    print(get_rec_lattice([[1,0,0], [0,1,0], [0,0,1]]) @ np.array([0,0,0.1]))

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

# def calculate_spectrum(bands, position):
#     omega = 


# dir_path = "alex_params"
# dir_path = "pbc_params"
# dir_path = "pbc_cutoff_corrected"
# read seedname_tb.dat with read_tb -> realspace matrices for different lattice vectors
# use Hk or Dk to convert realspace to reciprocal space matrices for given k-point 
# get k-points of interest with parsePath
# for each k point calculate r in orthogonal basis by löwdin othogonalization
# lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb(dir_path+"/seedname_tb.dat")

lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb("seedname_tb.dat")
get_rec_lattice(lattice)


MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
             'M' : np.array([0.5, 0, 0]),
             'K' : np.array([1/3, 1/3, 0]),
            }
segments, labels, bands1, bands2, Rk_orth, S_eigvals =  orthogonalize_basis(lattice, cells, degeneracy, Hr, Sr, Rr)
# print(np.shape(Rk_orth))
# print(np.shape(bands1))
# segments, labels = parsePath("GMKG", lattice, labelToK=MoS2_labelToK)

# vals_dftb = parse_dftb_band(dir_path+"/band.out", 8)
# vals_dftb = [vals_dftb[:100], vals_dftb[100:200], vals_dftb[200:300]]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
scale = 800
for i, (kPoints, relPos) in enumerate(segments):
    # Hk = w90.Hk(cells=cells,H=Hr, kFrac=kPoints)
    # Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
    # vals, vecs = sp.linalg.eig(Hk, Sk)
    # ax.plot(relPos, vals_dftb[i], '.', color="blue", ms=1)
    # ax.plot(relPos, vals, '.', color='orange', ms=1)
    # ax.plot(relPos, bands1[i], '.', color='blue',ms=1)
    # ax.plot(relPos, bands2[i], '.', color='orange', ms=1)
    # ax.plot(relPos, S_eigvals[i], '.', color='orange', ms=1)
    Rk = Rk_orth[i]
    # print(np.shape(Rk[:,0,0,0]))
    ax.plot(relPos, Rk[:,0,0,0], '.', color='orange', ms=1)
    ax.plot(relPos, Rk[:,0,0,1], '.', color='blue',  ms=1)
    # ax.plot(relPos, Rk[:,0,0,2], '.', color='green', ms=1)

fig.tight_layout()
l, pos = zip(*labels)
orthogonalize_basis(lattice, cells, degeneracy, Hr, Sr, Rr)
point_symbols = []
for i in labels:
    point_symbols.append(i[0])
plotLines(ax=ax, pos=pos, labels=point_symbols)
plt.show()

