import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'


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
    ax.set_ylabel(r"$E$ [eV]", labelpad=-5)

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

def k_grid_bz(lattice, shape:tuple, Gmax=1):
    rec_lat = get_rec_lattice(lattice=lattice).T
    n1, n2, n3 = shape[0], shape[1], shape[2]
    grids = [
        np.linspace(-0.5, 0.5, n1, endpoint=False),
        np.linspace(-0.5, 0.5, n2, endpoint=False),
        np.linspace(-0.5, 0.5, n3, endpoint=False),
    ]
    frac_k = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    k_cart = frac_k @ rec_lat
    G_frac = np.stack(
        np.meshgrid(
            range(-Gmax, Gmax + 1),
            range(-Gmax, Gmax + 1),
            range(-Gmax, Gmax + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    G_frac = G_frac[np.any(G_frac != 0, axis=1)]
    G_cart = G_frac @ rec_lat
    inside_mask = []
    for k in k_cart:
        inside_mask.append(
            np.all(np.linalg.norm(k) <= np.linalg.norm(k - G_cart, axis=1))
        )
    inside_mask = np.array(inside_mask)
    return frac_k[inside_mask]

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

def fermi_dirac(T_K, E_state_au, mu_au):
    mu_eV = au_to_eV(mu_au)
    E_state_eV = au_to_eV(E_state_au)
    kb = sc.constants.physical_constants['Boltzmann constant in eV/K'][0]
    if T_K == 0:
        f = np.zeros_like(E_state_eV, dtype=float)
        f[E_state_eV < mu_eV] = 1.0
        f[E_state_eV == mu_eV] = 0.5
        return f
    f = 1/(np.exp((E_state_eV - mu_eV )/ (T_K * kb))+1)
    return f

def fermi_dirac_dE(T_K, E_state_au, mu_au):
    E_state_eV = au_to_eV(E_state_au)
    mu_eV = au_to_eV(mu_au)
    kb = sc.constants.physical_constants['Boltzmann constant in eV/K'][0]
    if T_K == 0:
        return np.zeros_like(E_state_eV, dtype=float)
    x = (E_state_eV - mu_eV) / (kb * T_K)
    f_eV = -1.0 / (4 * kb * T_K * np.cosh(x / 2)**2)  
    f_au = f_eV * au_to_eV(1.0)
    return f_au

#TODO: function to calculate chemical potential

def check_vector_hermitian(pk, atol):
    is_hermitian = True
    for i in range(3):
        p = pk[:,:,:,i]
        if not np.all(sc.linalg.ishermitian(p, atol=atol)):
            is_hermitian = False
    return is_hermitian

def plot_bands(segments, labels, legend, bandstructures:list, pltname):
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    scale = 800
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for j, bands in enumerate(bandstructures):
        ax.plot([], [], color=colors[names[j]], label=legend[j])
        for i, (kPoints, relPos) in enumerate(segments[j]):
            ax.plot(relPos, bands[i], 
                    # '.',
                      color=colors[names[j]],ms=1)
    ax.set_ylim(bottom=-12, top=-1.5)
    fig.tight_layout()
    l, pos = zip(*labels)
    point_symbols = []
    for i in labels:
        point_symbols.append(i[0])
    plotLines(ax=ax, pos=pos, 
            #   labels=point_symbols,
            labels=[r"$\Gamma$", point_symbols[1], point_symbols[2], r"$\Gamma$"]
              )
    ax.legend(loc='best')
    plt.savefig(f'{pltname}.pdf')
    plt.show()