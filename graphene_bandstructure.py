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
    bands1 = []
    bands2 = []
    for i, (kPoints, relPos) in enumerate(segments):
        Hk = w90.Hk(cells=cells,H=Hr, kFrac=kPoints)
        Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
        # for ki in range(len(kPoints)):
        #     print(sc.linalg.ishermitian(Hk[ki]))
        #     print(sc.linalg.ishermitian(Sk[ki]))
        Rk = w90.Dk(cells=cells, D=Rr, kFrac=kPoints)
        # S_minushalf = loewdin_orth(Sk)
        S_minushalf = svd_inverse_sqrt(Sk)
        S_minushalf_partial = partial_Sminushalf(Sr, kPoints=kPoints)
        Hk_orth = S_minushalf @ Hk @ S_minushalf
        S_minushalf_dagger = np.transpose(S_minushalf.conj(), axes=(0,2,1))
        Rk_orth = np.einsum('kab,kbc,kcd,z->kadz', S_minushalf_dagger, Sk, S_minushalf_partial, np.ones(3)) + np.einsum('kab,kbcd,kce->kaed',S_minushalf_dagger, Rk, S_minushalf)
        for ki in range(np.shape(Hk_orth)[0]):
            # print(sc.linalg.ishermitian(S_minushalf[ki]))
            # print(sc.linalg.ishermitian(Hk_orth[ki]))
            print(np.allclose(Sk[ki] @ S_minushalf[ki] @ S_minushalf[ki], np.eye(8)))
        vals, vecs = sc.linalg.eigh(Hk_orth)
        vals1, vecs1 = sc.linalg.eig(Hk, Sk)
        bands1.append(np.real(vals))
        bands2.append(np.real(vals1))
    return segments, labels, bands1, bands2


def partial_Sminushalf(Sr, kPoints):
    stencil_shift_factor = 0.001
    stencil_shift = stencil_shift_factor * (kPoints[1] - kPoints[0])
    kPoints_plus = kPoints + stencil_shift 
    kPoints_minus = kPoints - stencil_shift
    kPoints_2plus = kPoints + 2 * stencil_shift
    kPoints_2minus = kPoints - 2 * stencil_shift
    S_plus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_plus)
    S_minus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_minus)
    S_2plus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_2plus)
    S_2minus = w90.Hk(cells=cells, H=Sr, kFrac=kPoints_2minus)
    Shalf_plus = svd_inverse_sqrt(S_plus)
    Shalf_minus = svd_inverse_sqrt(S_minus)
    Shalf_2plus = svd_inverse_sqrt(S_2plus)
    Shalf_2minus = svd_inverse_sqrt(S_2minus)
    derivative = (8 * Shalf_plus - 8 * Shalf_minus + Shalf_2minus - Shalf_2plus)/(12 * np.linalg.norm(stencil_shift, axis=-1))
    return derivative



def loewdin_orth(Sk):
    vals, vecs = sc.linalg.eig(Sk)
    U = vecs
    Udagger = np.transpose(vecs.conj(), axes=(0,2,1))
    minus_half = 1/np.sqrt(vals) 
    N,M = vals.shape
    diags_minushalf = np.zeros((N,M,M))
    idx = np.arange(M)
    diags_minushalf[:,idx, idx] = minus_half
    S_minushalf = np.einsum('kab, kbc, kcd -> kad', Udagger, diags_minushalf, U)
    return S_minushalf

def svd_inverse_sqrt(Sk):
    U, s, Vh = np.linalg.svd(Sk)
    # Build σ^{-1/2} for each k
    inv_s_half = np.zeros_like(Sk)
    idx = np.arange(Sk.shape[1])
    inv_s_half[:, idx, idx] = 1 / np.sqrt(s)
    # S^{-1/2} = V Σ^{-1/2} U†   (general case)
    return Vh.conj().transpose(0,2,1) @ inv_s_half @ U.conj().transpose(0,2,1)

def calculate_spectrum(bands, position):
    omega = 


dir_path = "alex_params"
# dir_path = "pbc_params"
# dir_path = "pbc_cutoff_corrected"
# read seedname_tb.dat with read_tb -> realspace matrices for different lattice vectors
# use Hk or Dk to convert realspace to reciprocal space matrices for given k-point 
# get k-points of interest with parsePath
# for each k point calculate r in orthogonal basis by löwdin othogonalization
lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb(dir_path+"/seedname_tb.dat")
MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
             'M' : np.array([0.5, 0, 0]),
             'K' : np.array([1/3, 1/3, 0]),
            }
segments, labels, bands1, bands2 =  orthogonalize_basis(lattice, cells, degeneracy, Hr, Sr, Rr)
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
    ax.plot(relPos, bands1[i], '.', color='blue',ms=1)
    ax.plot(relPos, bands2[i], '.', color='orange', ms=1)
fig.tight_layout()
l, pos = zip(*labels)
orthogonalize_basis(lattice, cells, degeneracy, Hr, Sr, Rr)
point_symbols = []
for i in labels:
    point_symbols.append(i[0])
plotLines(ax=ax, pos=pos, labels=point_symbols)
plt.show()

