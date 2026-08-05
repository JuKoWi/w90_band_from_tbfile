"""Create colormap plots of different k-dependent quantities in the first Brillouin zone"""

from tb_calculations.extract_observables import get_dipole_orth, get_rec_lattice, get_momentum_orth, to_bloch_basis, get_dipole_atomic, get_momentum_bloch_realspace, get_momentum_bloch_lee, get_momentum_bloch, get_momentum_atomic, get_velocity_atomic
from tb_calculations.utils import k_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import tb_calculations.parse_and_FT as parse_and_FT

plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

def plot_scalar_bz(grid_shape, idxa, idxb, filename, fromFile):
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

def plot_vector_bz(grid_shape, idxa, idxb, filename, fromFile):
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

def compare_momenta_bz(grid_shape, idxa, idxb, filename_r, fromFile, component, logarithmic):
    lattice1, cells1, degeneracies1, Hr1, Sr1, Rr1 = parse_and_FT.read_tb(filename_r)
    # lattice2, cells2, degeneracies2, Hr2, Sr2, pr2 = parse_and_FT.read_tb_momentum(filename_p)

    rec_lat = get_rec_lattice(lattice=lattice1)
    print(rec_lat[:,0])
    print(rec_lat[:,1])
    x = np.linspace(start=-1, stop=1, num=grid_shape[0])
    y = np.linspace(start=-1, stop=1, num=grid_shape[1])
    z = np.linspace(start=-1, stop=1, num=grid_shape[2])
    xgrid, ygrid, zgrid = np.meshgrid(x,y,z, indexing='ij') 
    xlist = xgrid.flatten(order='C')
    ylist = ygrid.flatten(order='C')
    zlist = zgrid.flatten(order='C')
    kFrac = np.column_stack((xlist,ylist,zlist))
    filename_r = 'momentum_lee.npy'
    # filename_p = 'momentum_realspace.npy'
    if fromFile:
        vec_matrix_r = np.load(file=filename_r)
        # vec_matrix_p = np.load(file=filename_p)
    else:
        vec_matrix_r = get_momentum_bloch_lee(Hr=Hr1, Sr=Sr1, Rr=Rr1, degeneracies=degeneracies1, kPoints=kFrac, cells=cells1, lattice=lattice1)
        # vec_matrix_r = get_velocity_atomic(Hr=Hr1, Sr=Sr1, Rr=Rr1, degeneracies=degeneracies1, kFrac=kFrac, cells=cells1, lattice=lattice1)
        # vec_matrix_p = get_momentum_bloch_realspace(lattice=lattice2, cells=cells2, degeneracies=degeneracies2, Hr=Hr2, Sr=Sr2, pr=pr2, kPoints=kFrac)
        # vec_matrix_p = get_momentum_atomic(lattice=lattice2, cells=cells2, degeneracies=degeneracies2, Hr=Hr2, Sr=Sr2, pr=pr2, kFrac=kFrac)
        np.save(file=filename_r, arr=vec_matrix_r)
        # np.save(file=filename_p, arr=vec_matrix_p)

    k_r, m_r, n_r, c_r = np.shape(vec_matrix_r)
    vec_matrix_r = np.reshape(vec_matrix_r, shape=(grid_shape[0], grid_shape[1], grid_shape[2], m_r, n_r, c_r))
    vec_matrix_r_re = np.real(vec_matrix_r[:,:,0,idxa, idxb,:])
    vec_matrix_r_im = np.imag(vec_matrix_r[:,:,0,idxa, idxb,:])

    # k_p, m_p, n_p, c_p = np.shape(vec_matrix_p)
    # vec_matrix_p = np.reshape(vec_matrix_p, shape=(grid_shape[0], grid_shape[1], grid_shape[2], m_p, n_p, c_p))
    # vec_matrix_p_re = np.real(vec_matrix_p[:,:,0,idxa, idxb,:])
    # vec_matrix_p_im = np.imag(vec_matrix_p[:,:,0,idxa, idxb,:])
    # print(np.max(vec_matrix_p_im))

    X = rec_lat[0,0] * xgrid + rec_lat[0,1] * ygrid
    Y = rec_lat[1,0] * xgrid + rec_lat[1,1] * ygrid
    fig, axs = plt.subplots(ncols=2, figsize=(15,4.5))
    all_data = np.array([
        vec_matrix_r_re[..., component],
        vec_matrix_r_im[..., component],
        # vec_matrix_p_re[..., component],
        # vec_matrix_p_im[..., component]
    ])

    if logarithmic:
        all_data = np.abs(all_data) 
        all_data[all_data == 0] = 1e-14

    vmin = np.min(all_data)
    vmax = np.max(all_data)

    if logarithmic:
        norm=LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=Normalize(vmin=vmin, vmax=vmax)
        

    
    p1 = axs[0].pcolormesh(X[:,:,0], Y[:,:,0],
                            all_data[0],
                            # shading='gouraud',
                            edgecolors=None,
                            linewidth=0,
                            rasterized=True,
                            # norm=norm
                             )
    p2 = axs[1].pcolormesh(X[:,:,0], Y[:,:,0], 
                            all_data[1],
                            # shading='gouraud',
                            edgecolors=None,
                            linewidth=0,
                            rasterized=True,
                            # norm=norm
                             )
    # p3 = axs[1,0].pcolormesh(X[:,:,0], Y[:,:,0], 
    #                         all_data[2],
    #                         # shading='gouraud',
    #                         edgecolors=None,
    #                         linewidth=0,
    #                         rasterized=True,
    #                         # norm=norm
    #                          )
    # p4 = axs[1,1].pcolormesh(X[:,:,0], Y[:,:,0], 
    #                         all_data[3],
    #                         # shading='gouraud',
    #                         edgecolors=None,
    #                         linewidth=0,
    #                         rasterized=True,
    #                         # norm=norm
    #                          )
    fig.colorbar(p1, ax=axs[0])
    fig.colorbar(p2, ax=axs[1])
    # fig.colorbar(p3, ax=axs[1,0])
    # fig.colorbar(p4, ax=axs[1,1])

    for a in axs.flatten():
        a.set_aspect('equal')
    
    component_dict = {0: 'x', 1:'y', 2: 'z'}


    axs[0].set_title(fr'Re($v_{component_dict[component]}$)')
    axs[1].set_title(fr'Im($v_{component_dict[component]}$)')
    # axs[1,0].set_title(fr'Re($p_{component_dict[component]}$)') 
    # axs[1,1].set_title(fr'Im($p_{component_dict[component]}$)')
    axs[0].set_xlabel(r'$k_x$ $[\AA^{-1}]$')
    axs[1].set_xlabel(r'$k_x$ $[\AA^{-1}]$')
    # axs[1,0].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    # axs[1,1].set_xlabel(r'$k_x$ / $\AA^{-1}$')
    axs[0].set_ylabel(r'$k_y$ $[\AA^{-1}]$')
    axs[1].set_ylabel(r'$k_y$ $[\AA^{-1}]$')
    # axs[1,0].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    # axs[1,1].set_ylabel(r'$k_y$ / $\AA^{-1}$')
    plt.savefig(f"velocity_bz_{idxa}-{idxb}.pdf")
    plt.show()

# carbon_structure = 'cube/'
# seedname_dir = './seedname_pseudocarb_noshift/'
# filename_r = seedname_dir + carbon_structure + 'seedname_tb.dat'
# filename_p = seedname_dir + carbon_structure + 'seedname_tb_momentum.dat'


seedname_dir = './seedname_mos2/'
filename_r = seedname_dir + 'seedname_tb.dat'
filename_p = seedname_dir + 'seedname_tb_momentum.dat'

# plot_dipole_bz(grid_shape=(50, 50, 1), idxa=0, idxb=10, filename=fileA, fromFile=True)
# plot_scalar_bz(grid_shape=(50,50,1), idxa=0, idxb=10, filename=fileA, fromFile=False)
compare_momenta_bz(grid_shape=(50, 50,1), 
                   idxa=8, 
                   idxb=9, 
                    filename_r=filename_r,
                   fromFile=True,
                   component=1,
                   logarithmic=False)
