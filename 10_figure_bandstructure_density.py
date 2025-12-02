#!/usr/bin/env python3

import numpy as np
import pylab as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import ScaledTranslation

from pathlib import Path
import sys
plt.style.use('../plot_helper/wannierhousten.mplstyle')
sys.path.insert(0, str(Path.cwd().parent.absolute()))
from plot_helper import enable_latex, create_rectangle, save_pdf, place_labels
import w90




def drawExpandedRect(rax, xlim, ylim, expand=20):
    def pnd(lim):
        mid = 0.5 * (lim[0] + lim[1])
        newR = 0.5 * expand * (lim[1] - lim[0])
        return [mid - newR, mid + newR]
    xlimE = pnd(xlim)
    ylimE = pnd(ylim)
    create_rectangle(rax, (xlimE, ylimE), linewidth=0.5, facecolor='gray', alpha=0.2)
    create_rectangle(rax, (xlimE, ylimE), linewidth=0.5)


# sc=0.85
sc = 2
fig = plt.figure(figsize=(sc*4.2, sc*5))
gs = GridSpec(4, 3, figure=fig, height_ratios=(1.5, 1, 1, 0.1),
                                width_ratios=(1, 0.08, 1), wspace=0.03)



axBSfull = fig.add_subplot(gs[0, 0])
axBSdetail = fig.add_subplot(gs[0, 2])

axDMfull = fig.add_subplot(gs[1, 0])
axDMdetail = fig.add_subplot(gs[1, 2])

axClose = fig.add_subplot(gs[2, 0])
axClose.axis('off')
gs_close = gs[2, 0].subgridspec(1, 5, width_ratios=(0.2, 1, 0.03, 1, 0.01))
axClose_H = fig.add_subplot(gs_close[1])
axClose_W = fig.add_subplot(gs_close[3])

axMix = fig.add_subplot(gs[2, 2])
axMix.axis('off')
gs_mix = gs[2, 2].subgridspec(1, 5, width_ratios=(0.2, 1, 0.03, 1, 0.01))
axMix_H = fig.add_subplot(gs_mix[1])
axMix_W = fig.add_subplot(gs_mix[3])



axcbar_H = fig.add_subplot(gs[3, 0])
axcbar_W = fig.add_subplot(gs[3, 2])



color1 = '#d95f02'
color2 = '#1b9e77'
color3 = '#7570b3'
color4 = '#a19408'
color_gray = '#333333'

markers = ['o', 'h', 'D']

def plotScatter(cax, kz, cells, Hr, nScatter=20, scale=220):
    sw = kz.size // nScatter
    Hk = np.array([w90.Hk(cells, Hr, [0, 0, k]) for k in kz])
    E, U = np.linalg.eigh(Hk)

    contrib = np.abs(U * U.conj())

    for bi in [1, 2, 3]:
        cax.scatter(kz[::sw], E[::sw, bi], s=scale * contrib[::sw, bi, 3],
                   marker=markers[0], facecolor=color1, edgecolor=color1, alpha=0.7, zorder=10)
        cax.scatter(kz[sw // 3::sw], E[sw // 3::sw, bi], s=scale * contrib[sw // 3::sw, bi, 1],
                   marker=markers[1], facecolor=color2, edgecolor=color2, alpha=0.7, zorder=10)
        cax.scatter(kz[(2 * sw) // 3::sw], E[(2 * sw) // 3::sw, bi], s=scale * contrib[(2 * sw) // 3::sw, bi, 2],
                   marker=markers[2], facecolor=color3, edgecolor=color3, alpha=0.7, zorder=10)
    return E


def create_dummy_legend(cax):
    xlim = cax.get_xlim()
    ylim = cax.get_ylim()
    cax.scatter([-10], [-10], marker=markers[0], facecolor=color1, edgecolor=color1, alpha=0.7, zorder=10, label='WF 2')
    cax.scatter([-10], [-10], marker=markers[1], facecolor=color2, edgecolor=color2, alpha=0.7, zorder=10, label='WF 3')
    cax.scatter([-10], [-10], marker=markers[2], facecolor=color3, edgecolor=color3, alpha=0.7, zorder=10, label='WF 4')
    cax.legend(fontsize=13, loc=(0.59, 0.60), handlelength=0.3, scatteryoffsets=[0.5])
    cax.set_xlim(xlim)
    cax.set_ylim(ylim)

lattice, cells, Hr, Rr = w90.read_wsvectb("CdSe_9x9x9_qe_cb2_vb6")

# shift Fermi energy to 0:
HGamma = w90.Hk(cells, Hr, [0, 0, 0])
EGamma = np.linalg.eigvalsh(HGamma)
Ef = 0.5 * (EGamma[5] + EGamma[6])
Hr[cells.index((0, 0, 0))] -= np.diag(EGamma.size * [Ef] )

full_kz_lim = [-0.025, 0.525]
detail_kz_lim = [0.1445, 0.1485]



kz = np.linspace(full_kz_lim[0], full_kz_lim[1], 500, endpoint=True)
E = plotScatter(axBSfull, kz, cells, Hr)
axBSfull.plot(kz, E, color=color_gray)
create_dummy_legend(axBSfull)
axBSfull.set_xlim(full_kz_lim)
axBSfull.set_ylabel(r"$E$ [eV]", labelpad=10)
axBSfull.xaxis.set_ticklabels([])

kz = np.linspace(detail_kz_lim[0], detail_kz_lim[1], 500, endpoint=True)
E = plotScatter(axBSdetail, kz, cells, Hr)
axBSdetail.plot(kz, E[:, 1:4], color=color_gray)
axBSdetail.set_xlim(detail_kz_lim)
axBSdetail.set_yticks([-1.52, -1.53])
axBSdetail.yaxis.set_label_position("right")
axBSdetail.yaxis.tick_right()

axBSdetail.xaxis.set_ticklabels([])

drawExpandedRect(axBSfull, axBSdetail.get_xlim(), axBSdetail.get_ylim())


data = np.load('data_dephasing_10000.npz')
T2_raw_slice1 = data['T2_raw_slice1'][::-1]
T2_raw_slice2 = data['T2_raw_slice2'][::-1]
vminH = data['vminH']
vmaxH = data['vmaxH']
vminW = data['vminW']
vmaxW = data['vmaxW']
rho_mixH = data['rho_mixH']
rho_mixW = data['rho_mixW']
rho_closeH = data['rho_closeH']
rho_closeW = data['rho_closeW']

index_close = 8533
index_mix = 8544
kzOrig = data['kz'] / data['kz'].size

kz = 1 - kzOrig[::-1]
kz_mix = 1 - kzOrig[index_mix]
kz_close = 1 - kzOrig[index_close]

def plot_lines(ax, s=0, label=None):
    if not label:
        label = [None, None]
    ax.plot(kz + s, T2_raw_slice1, lw=2, color=color1, label=label[0])
    ax.plot(kz + s, T2_raw_slice2, lw=2, color=color4, label=label[1])

plot_lines(axDMfull, s=0, label=[r"$(m, n)=(2, 2)$", r"$(m, n)=(2, 7)$"])
plot_lines(axDMfull, s=-1)
plot_lines(axDMdetail, s=0)

axDMdetail.set_xlim(detail_kz_lim)
axDMdetail.set_xticks([0.145, 0.148])
axDMdetail.set_ylim([axDMdetail.get_ylim()[0], 0.00028])

axDMfull.legend(fontsize=14)
axDMfull.set_xlim(full_kz_lim)
axDMfull.set_ylim(axDMdetail.get_ylim())

# After creating the plot and before saving
axDMfull.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axDMfull.yaxis.get_offset_text().set_fontsize(14)  # Optional: adjust the size of the exponent
# axDMfull.yaxis.set_offset_position('right')
# Or for more control, use a ScalarFormatter:
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))  # Force scientific notation
axDMfull.yaxis.set_major_formatter(formatter)


axDMfull.set_ylabel(r'$\left|(\partial_t\rho^{\mathrm{W}}_{mn})_{\mathrm{Deph}}\right|$')

axDMdetail.set_yticklabels([])
axDMdetail.set_xlabel(r"$k_z$ $\left[\frac{2\pi}{c}\right]$", labelpad=-10)



axDMfull.set_xlabel(r"$k_z$ $\left[\frac{2\pi}{c}\right]$", labelpad=-28) # -11.5
axDMfull.set_xticks([0, 0.25, 0.5])
axDMfull.set_xticklabels([r'\parbox{2cm}{\centering 0\\ $\Gamma$}', '',
                          r'\parbox{2cm}{\centering 0.5\\A}'])
axDMfull.set_xlim(full_kz_lim)

drawExpandedRect(axDMfull, axDMdetail.get_xlim(), axDMdetail.get_ylim())

for axl in [axDMdetail, axBSdetail]:
    axl.axvline(kz_mix, color='k', ls='--')
    axl.axvline(kz_close, color='k', ls='--')

kz_d = detail_kz_lim[1] - detail_kz_lim[0]
axDMdetail.text(kz_mix - 0.02 * kz_d, 0.88 * axDMdetail.get_ylim()[1], r"$\mathrm{P}$", ha='right')
axDMdetail.text(kz_close - 0.02 * kz_d, 0.88 * axDMdetail.get_ylim()[1], r"$\mathrm{Q}$", ha='right')

def showDensity(dax, rho, lnorm, lcmap, yticks=False):
    X, Y = np.mgrid[1:rho_mixH.shape[0]+1, 1:rho_mixH.shape[1]+1]
    mesh = dax.pcolormesh(X, Y, rho, shading='nearest', cmap=lcmap, norm=lnorm)
    dax.set_aspect('equal')
    dax.axhline(6.5, color='gray', lw=0.5, ls='--')
    dax.axvline(6.5, color='gray', lw=0.5, ls='--')
    dax.set_xticks([1, 6])
    if  yticks:
        dax.set_yticks([1, 6])
    else:
        dax.set_yticks([])
    return mesh


def squeeze_map(cmap: str):
    clrs = plt.get_cmap(cmap)(np.linspace(0, 1, 256) ** (1 / 2.))
    cmap = LinearSegmentedColormap.from_list('test', clrs)
    cmap.set_bad('#FFFFFF')
    return cmap

norm_H = Normalize(vmin=vminH, vmax=vmaxH)
norm_W = Normalize(vmin=vminW, vmax=vmaxW)
cmap_H = squeeze_map('bone_r')
cmap_W = squeeze_map('pink_r')

showDensity(axMix_H, rho_mixH, norm_H, cmap_H, yticks=True)
showDensity(axMix_W, rho_mixW, norm_W, cmap_W)

showDensity(axClose_H, rho_closeH, norm_H, cmap_H, yticks=True)
showDensity(axClose_W, rho_closeW, norm_W, cmap_W)



def drawBox(fig, axes, names, **kwargs):
    wpad, hpad = 0, 0.03
    bbox = axes[0].get_tightbbox(fig.canvas.get_renderer())
    xmin, ymin, xmax, ymax = bbox.transformed(fig.transFigure.inverted()).extents
    for bax in axes[1:]:
        bbox = bax.get_tightbbox(fig.canvas.get_renderer())
        cxmin, cymin, cxmax, cymax = bbox.transformed(fig.transFigure.inverted()).extents
        xmin = min(xmin, cxmin)
        ymin = min(ymin, cymin)
        xmax = max(xmax, cxmax)
        ymax = max(ymax, cymax)
    fig.add_artist(FancyBboxPatch((xmin-wpad/2, ymin-hpad), xmax-xmin+wpad, ymax-ymin+hpad,
                                   boxstyle="Round,pad=0.012", **kwargs))
    for nax, name in zip(axes, names):
        nax.text(4.75, -3.3, name, ha='center')


cbar_W = fig.colorbar(plt.cm.ScalarMappable(norm=norm_W, cmap=cmap_W), cax=axcbar_W, orientation='horizontal')
cbar_W.set_label( r'$\left|(\partial_t\rho^{\mathrm{W}})_{\mathrm{Deph}}\right|$')
cbar_W.set_ticks([0, 1e-4, 2e-4])
cbar_W.set_ticklabels([r'$0$', r'$1\cdot 10^{-4}$', r'$2\cdot 10^{-4}$'])
# axcbar_W.xaxis.set_major_formatter(formatter)
cbar_H = fig.colorbar(plt.cm.ScalarMappable(norm=norm_H, cmap=cmap_H), cax=axcbar_H, orientation='horizontal')
# cbar_H.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar_H.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
cbar_H.set_label(r'$|\rho^{\mathrm{H}}|$')


# fig.tight_layout()
fig.subplots_adjust(top=0.99, left=0.10, right=0.91, hspace=0.15, bottom=0.08)

drawBox(fig, [axMix_H, axMix_W],
             names=[r"$|\rho^{\mathrm{H}}_{\mathrm{Q}}|$",
                    r"$|(\partial_t\rho^{\mathrm{W}}_{\mathrm{Q}})_{\mathrm{Deph}}|$"],
             edgecolor='gray', linewidth=3, fill=False)
drawBox(fig, [axClose_H, axClose_W],
             names=[r"$|\rho^{\mathrm{H}}_{\mathrm{P}}|$",
                    r"$|(\partial_t\rho^{\mathrm{W}}_{\mathrm{P}})_{\mathrm{Deph}}|$"],
             edgecolor='gray', linewidth=3, fill=False)

place_labels(fig, np.array([axBSfull, axBSdetail, axDMfull, axDMdetail]))

for ax, label in [ (axClose,"e)"), (axMix, "f)") ]:
    trans = ScaledTranslation(10/72, -33/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.2'),
            zorder=1000)


filename = 'bandstructure_density.pdf'
save_pdf(fig, filename)
plt.show()
