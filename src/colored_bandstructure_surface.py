from tb_calculations.extract_observables import (bandstructure_from_k,
                                                get_rec_lattice,
                                                velocity_bloch_lee,
                                                 )
from tb_calculations.parse_and_FT import (read_tb,
                                          Hk_degenerate
                                          )
from tb_calculations.utils import (k_grid, 
                                   k_grid_bz,
                                    au_to_eV
                                    )
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy as sc
from scipy.spatial import Voronoi

MARGIN = 0.25          # fraction of |K| drawn beyond the zone edge


def bz_polygon(rec_lat):
    """Vertices of the 2D Wigner-Seitz cell of the reciprocal lattice, angle-sorted."""
    b1, b2 = rec_lat[:2, 0], rec_lat[:2, 1]
    pts = np.array([i * b1 + j * b2 for i in range(-2, 3) for j in range(-2, 3)])
    vor = Voronoi(pts)
    region = vor.regions[vor.point_region[np.argmin(np.linalg.norm(pts, axis=1))]]
    verts = vor.vertices[region]
    return verts[np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))]


def fold_with_margin(kfrac, rec_lat, r_cut):
    """Cartesian k of every periodic image inside r_cut, plus index into the original list."""
    f = kfrac - np.rint(kfrac)
    shifts = np.array([[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
    cand = (f[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
    idx = np.repeat(np.arange(len(f)), len(shifts))
    cart = cand @ rec_lat.T
    keep = np.hypot(cart[:, 0], cart[:, 1]) <= r_cut
    cart, idx = cart[keep], idx[keep]
    _, uniq = np.unique(np.round(cart[:, :2], 8), axis=0, return_index=True)
    return cart[uniq], idx[uniq]


grid_shape = (50, 50, 1)          # divisible by 3 so K is on the grid
vb_idx = 8
lattice, cells, degeneracies, Hr, Sr, Rr = read_tb("./data/seedname_mos2/seedname_tb.dat")
rec_lat = get_rec_lattice(lattice=lattice)
x = np.linspace(start=0, stop=1, num=grid_shape[0], endpoint=False)
y = np.linspace(start=0, stop=1, num=grid_shape[1], endpoint=False)
z = np.linspace(start=0, stop=1, num=grid_shape[2], endpoint=False)
xgrid, ygrid, zgrid = np.meshgrid(x, y, z, indexing='ij')
xlist = xgrid.flatten(order='C')
ylist = ygrid.flatten(order='C')
zlist = zgrid.flatten(order='C')
kfrac = np.column_stack((xlist, ylist, zlist))
Hk_atomic = Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Hr, kFrac=kfrac)
Sk_atomic = Hk_degenerate(cells=cells, degeneracies=degeneracies, Hr=Sr, kFrac=kfrac)
eigvals, U = sc.linalg.eigh(Hk_atomic, Sk_atomic)
eigvals = au_to_eV(eigvals)
vb = eigvals[:, vb_idx].copy()
cb = eigvals[:, vb_idx+1].copy()
E_fermi = 0.5 *(vb.max() + cb.min())
vb -= E_fermi
cb -= E_fermi
E_fermi = 0
velocity = velocity_bloch_lee(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kfrac, cells=cells, lattice_au=lattice) # (k,a,b,z)
velocity = np.sqrt(np.real(np.einsum('kabz, kabz->kab',velocity.conj()[...,:2], velocity[...,:2])))[:,vb_idx, vb_idx+1]


hexagon = bz_polygon(rec_lat)
r_K = np.max(np.linalg.norm(hexagon, axis=1))
kcart, idx = fold_with_margin(kfrac, rec_lat, r_cut=r_K * (1.0 + MARGIN))
X, Y = kcart[:, 0], kcart[:, 1]

tri = mtri.Triangulation(X, Y)
edge = np.max([np.hypot(X[tri.triangles[:, i]] - X[tri.triangles[:, j]],
                        Y[tri.triangles[:, i]] - Y[tri.triangles[:, j]])
               for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
tri.set_mask(edge > 1.5 * np.linalg.norm(rec_lat[:2, 0]) / grid_shape[0])

quantity = velocity[idx]
faces = tri.get_masked_triangles()
facevals = quantity[faces].mean(axis=1)
norm = plt.Normalize(vmin=facevals.min(), vmax=facevals.max())
cmap = plt.get_cmap('gnuplot')

fig, ax = plt.subplots(figsize=(12,9), subplot_kw=dict(projection='3d', computed_zorder=False))

loop = np.vstack([hexagon, hexagon[:1]])
ax.plot(loop[:, 0], loop[:, 1], E_fermi, color='k', lw=1.5)
for vx, vy in hexagon:                      # verticals at the K points
    ax.plot([vx, vx], [vy, vy], [E_fermi, cb.max()], color='k', lw=0.6, alpha=0.4, zorder=2)
ax.scatter([0], [0], [E_fermi], color='k', s=15)
ax.text(0, 0, E_fermi, r'  $\Gamma$', size='x-small')
ax.text(hexagon[0, 0], hexagon[0, 1], E_fermi, '  K', size='x-small')
mid = 0.5 * (hexagon[0] + hexagon[1])
ax.text(mid[0], mid[1], E_fermi, '  M', size='x-small')

surf_vb = ax.plot_trisurf(tri, vb[idx], linewidth=0, shade=False, antialiased=False, zorder=1)
surf_vb.set_fc(cmap(norm(facevals)))
surf_cb = ax.plot_trisurf(tri, cb[idx], linewidth=0, shade=False, antialiased=False, zorder=3)
surf_cb.set_fc(cmap(norm(facevals)))
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(''); ax.set_ylabel('')
ax.set_zlabel('E [eV]', fontsize=11, 
              labelpad=8
              )
ax.set_xlabel(r'$k_x$', fontsize=11)
ax.set_ylabel(r'$k_y$', fontsize=11)
ax.tick_params(axis='z', labelsize=10)



mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap,)
mappable.set_array([])
cb_ = fig.colorbar(mappable, ax=ax, shrink=0.45, aspect=25, pad=0.05, panchor=(1.0,0.7))
# cb_.set_label(r'$||\mathbf{v}_\mathrm{vc}||$ [a.u.]', fontsize=11)
cb_.set_label(r'$\left|\langle c,\mathbf{k}|\hat{\mathbf{v}}|v,\mathbf{k}\rangle\right|$  [a.u.]', fontsize=11)
cb_.ax.tick_params(labelsize=9)


for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor('none')
ax.grid(False)


def level_plane(E, color, alpha=0.13, z=2):
    poly = np.column_stack([hexagon, np.full(len(hexagon), E)])
    ax.add_collection3d(Poly3DCollection([poly], facecolor=color, alpha=alpha,
                                         edgecolor='none', zorder=z))
    loop = np.vstack([poly, poly[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
            color=color, lw=1.0, ls=(0, (5, 3)), zorder=z)

C_VB, C_CB = '#1f77b4', '#d62728'

# level_plane(-1, C_VB)
# level_plane(1, C_CB)

E_vbm = np.max(vb)
E_cbm = np.min(cb)
x_wall, y_wall = -r_K * 1.05, r_K * 1.05     # xmin and ymax walls, correct for azim=-60
z_floor = vb.min() - 0.3 * (cb.max() - vb.min())
# ax.set_xlim(x_wall, r_K * 1.05)
# ax.set_ylim(-r_K * 1.05, y_wall)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

for E, c in ((E_vbm, C_VB), (E_cbm, C_CB)):
    ax.plot(xmin, [ymin,ymax], [E, E], color='grey', ls=(0, (5, 3)), lw=1.0, zorder=0)
    ax.plot([xmin, xmax], ymax, [E, E], color='grey', ls=(0, (5, 3)), lw=1.0, zorder=0)

ax.view_init(elev=15, azim=-60, roll=0)
fig.savefig("3d_band.pdf", bbox_inches=Bbox([[2,1.8],[10.5,6.5]]))
plt.show()