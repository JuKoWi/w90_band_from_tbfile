from src.from_tb.extract_properties_tbfile import get_dipole_orth, k_grid, get_rec_lattice, get_momentum_orth, to_bloch_basis, get_dipole_atomic
import numpy as np
import matplotlib.pyplot as plt
import from_tb.parse_and_FT as parse_and_FT

plt.rcParams.update({'font.size': 45})
plt.rcParams['savefig.bbox'] = 'tight'

def plot_overlap_bz(grid_shape, idxa, idxb, filename, fromFile):
    """plot the dipole operator elements in orthogonal basis in the brillouin zone"""
    lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(filename)
    rec_lat = get_rec_lattice(lattice=lattice)
    print(rec_lat[:,0])
    print(rec_lat[:,1])
    x = np.linspace(start=0, stop=1, num=grid_shape[0])
    y = np.linspace(start=0, stop=1, num=grid_shape[1])
    z = np.linspace(start=0, stop=1, num=grid_shape[2])
    xgrid, ygrid, zgrid = np.meshgrid(x,y,z, indexing='ij') 
    xlist = xgrid.flatten(order='C')
    ylist = ygrid.flatten(order='C')
    zlist = zgrid.flatten(order='C')
    kfrac = np.column_stack((xlist,ylist,zlist))

    filename = 'overlap_k_atomic.npy'
    if fromFile:
        matrix = np.load(file=filename)
    else:
        # Rk = w90.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kfrac)
        # dk_orth, Sk_orth, Hk_orth = get_dipole_orth(Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kfrac, cells=cells, lattice=lattice, degeneracies=degeneracies)
        # pk, Sk_orth, Hk_orth = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kfrac, cells=cells,lattice=lattice)
        # pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth, Sk_orth=Sk_orth)

        matrix = parse_and_FT.Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kfrac)
        np.save(file=filename, arr=matrix)

    k, m, n = np.shape(matrix)
    matrix = np.reshape(matrix, shape=(grid_shape[0], grid_shape[1], grid_shape[2], m, n))
    matrix_re = np.real(matrix[:,:,0,idxa, idxb])
    matrix_im = np.imag(matrix[:,:,0,idxa, idxb])

    X = rec_lat[0,0] * xgrid + rec_lat[0,1] * ygrid
    Y = rec_lat[1,0] * xgrid + rec_lat[1,1] * ygrid
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(30,9))
    p1 = axs[0].pcolormesh(X[:,:,0], Y[:,:,0], matrix_re, 
                            shading='gouraud'
                             )
    p2 = axs[1].pcolormesh(X[:,:,0], Y[:,:,0], matrix_im, 
                            shading='gouraud'
                             )
    axs[0].set_title(r'Re($S$)')
    axs[1].set_title(r'Im($S$)')
    axs[0].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[1].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[0].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    axs[1].set_ylabel(r'$k_y$ / $\AA^{-1}$')

    fig.colorbar(p1, ax=axs[0])
    fig.colorbar(p2, ax=axs[1])
    plt.savefig(f"overlap_{idxa}-{idxb}.pdf")
    plt.show()

def plot_dipole_bz(grid_shape, idxa, idxb, filename, fromFile):
    """plot the dipole operator elements in orthogonal basis in the brillouin zone"""
    lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(filename)
    rec_lat = get_rec_lattice(lattice=lattice)
    print(rec_lat[:,0])
    print(rec_lat[:,1])
    x = np.linspace(start=0, stop=1, num=grid_shape[0])
    y = np.linspace(start=0, stop=1, num=grid_shape[1])
    z = np.linspace(start=0, stop=1, num=grid_shape[2])
    xgrid, ygrid, zgrid = np.meshgrid(x,y,z, indexing='ij') 
    xlist = xgrid.flatten(order='C')
    ylist = ygrid.flatten(order='C')
    zlist = zgrid.flatten(order='C')
    kfrac = np.column_stack((xlist,ylist,zlist))

    filename = 'dipole_k_atomic.npy'
    if fromFile:
        vec_matrix = np.load(file=filename)
    else:
        # Rk = w90.Rk_degenerate(cells=cells, degeneracies=degeneracies, Rr=Rr, kFrac=kfrac)
        # dk_orth, Sk_orth, Hk_orth = get_dipole_orth(Hr=Hr, Sr=Sr, Rr=Rr, kPoints=kfrac, cells=cells, lattice=lattice, degeneracies=degeneracies)
        # pk, Sk_orth, Hk_orth = get_momentum(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kfrac, cells=cells,lattice=lattice)
        # pk_bloch, Hk_bloch, Sk_bloch = to_bloch_basis(pk=pk, Hk_orth=Hk_orth, Sk_orth=Sk_orth)
        dk_atomic = get_dipole_atomic(Rr=Rr, Sr=Sr, kPoints=kfrac, cells=cells, lattice=lattice, degeneracies=degeneracies)
        vec_matrix = dk_atomic 
        np.save(file=filename, arr=dk_atomic)

    k, m, n, c = np.shape(vec_matrix)
    vec_matrix = np.reshape(vec_matrix, shape=(grid_shape[0], grid_shape[1], grid_shape[2], m, n, c))
    vec_matrix_re = np.real(vec_matrix[:,:,0,idxa, idxb,:])
    vec_matrix_im = np.imag(vec_matrix[:,:,0,idxa, idxb,:])

    X = rec_lat[0,0] * xgrid + rec_lat[0,1] * ygrid
    Y = rec_lat[1,0] * xgrid + rec_lat[1,1] * ygrid
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(65,10))
    p1 = axs[0].pcolormesh(X[:,:,0], Y[:,:,0], vec_matrix_re[...,0], 
                            shading='gouraud'
                             )
    p2 = axs[1].pcolormesh(X[:,:,0], Y[:,:,0], vec_matrix_im[...,0], 
                            shading='gouraud'
                             )
    p3 = axs[2].pcolormesh(X[:,:,0], Y[:,:,0], vec_matrix_im[...,1],
                            shading='gouraud'
                             )
    p4 = axs[3].pcolormesh(X[:,:,0], Y[:,:,0], vec_matrix_im[...,1], 
                            shading='gouraud'
                             )
    axs[0].set_title(r'Re($d_x$)')
    axs[1].set_title(r'Im($d_x$)')
    axs[2].set_title(r'Re($d_y$)')
    axs[3].set_title(r'Im($d_y$)')
    axs[0].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[1].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[2].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[3].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[0].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    axs[1].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    axs[2].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    axs[3].set_ylabel(r'$k_y$ / $\AA^{-1}$')


    # axs[0,0].set_xlim(left=-1, right=1)
    # axs[0,0].set_ylim(top=0.6, bottom=-0.6)
    # axs[0,1].set_xlim(left=-1, right=1)
    # axs[0,1].set_ylim(top=0.6, bottom=-0.6)
    # axs[1,0].set_xlim(left=-1, right=1)
    # axs[1,0].set_ylim(top=0.6, bottom=-0.6)
    # axs[1,1].set_xlim(left=-1, right=1)
    # axs[1,1].set_ylim(top=0.6, bottom=-0.6)
    fig.colorbar(p1, ax=axs[0])
    fig.colorbar(p2, ax=axs[1])
    fig.colorbar(p3, ax=axs[2])
    fig.colorbar(p4, ax=axs[3])
    plt.savefig(f"dipole_{idxa}-{idxb}.pdf")
    plt.show()

fileA = 'seedname_mos2_ase.dat'
# plot_dipole_bz(grid_shape=(50, 50, 1), idxa=0, idxb=10, filename=fileA, fromFile=True)
plot_overlap_bz(grid_shape=(50,50,1), idxa=0, idxb=10, filename=fileA, fromFile=False)
