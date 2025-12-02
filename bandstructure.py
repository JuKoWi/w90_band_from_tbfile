#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# plt.style.use('../plot_helper/style.mplstyle')
sys.path.insert(0, str(Path.cwd().parent.absolute()))

from multiprocessing import Pool
# import psutil
# from threadpoolctl import threadpool_limits


# from plot_helper import atu, place_labels, save_pdf
# from KspaceInterpolator import KspaceInterpolator

from itertools import cycle


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


def plotMoS2():
    fname = "MoS2_10x10x1_tb.npz"
    ksi = KspaceInterpolator.from_file(fname)
    lattice = ksi.lattice
    segments, labels = parsePath("GMKG", lattice, labelToK) 

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scale = 800

    for i, (kPoints, relPos) in enumerate(segments):
            H = ksi.k_points("H", kPoints)
            Rorig = ksi.k_points('Rorig', kPoints)
            dHk = ksi.dk_points("H", kPoints)
            pCom = 1j *(  np.einsum("smk,sknd->smnd", H, Rorig)
                        - np.einsum("smkd,skn->smnd", Rorig, H)) + dHk
            p = ksi.k_points("p", kPoints) 
            p = ksi.k_points("p", kPoints) - pCom
            E, U = np.linalg.eigh(H)
            pH = np.einsum("sba,sbcx,scd->sadx", U.conj(), p, U, optimize=True)[...,0:2]
            E *= 27.21
            ax.plot(relPos, E, '.', color='orange', ms=1)
            for r in range(E.shape[1]):
                ax.scatter(relPos, E[:, r], s=scale * np.linalg.norm(pH[:, r]), color='k')
    fig.tight_layout()



def plotSilicon():
    fname = "silicon_tb.npz"
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    scale = 5e3
    l, pos = zip(*labels)
    # manually fix Labels
    ax.set_xticks(pos, l)
    ax.set_xlim([pos[0], pos[-1]])
    ax.set_ylabel("E [eV]")
    fig.tight_layout()

def plotPath(ax, fname, labelToK, path, **kwargs):
    ksi = KspaceInterpolator.from_file(fname)
    mc = -ksi.minIndices
    H = ksi.rMats['H']
    lattice = ksi.lattice
    segments, labels = parsePath(path, lattice, labelToK)
    for i, (kPoints, relPos) in enumerate(segments):
        H = ksi.k_points("H", kPoints)
        # Rorig = ksi.k_points('Rorig', kPoints)
        # dHk = ksi.dk_points("H", kPoints)
        #pCom = 1j *(  np.einsum("smk,sknd->smnd", H, Rorig)
        #            - np.einsum("smkd,skn->smnd", Rorig, H)) + dHk
        #p = ksi.k_points("p", kPoints) 
        #p = ksi.k_points("p", kPoints) - pCom
        with threadpool_limits(limits=1, user_api='openmp'):
            with Pool(psutil.cpu_count(logical=False)) as p:
                EList = p.map(np.linalg.eigvalsh, H)
        E = np.array(EList)
        # E, U = np.linalg.eigh(H)
        #pH = np.einsum("sba,sbcx,scd->sadx", U.conj(), p, U, optimize=True)
        #p = np.abs(pH[:, :, :, 0])
        E = atu.to_eV(E)
        for s in range(E.shape[1]):
            ax.plot(relPos, E[:, s], **kwargs)
            kwargs['label'] = None
    l, pos = zip(*labels)
    return pos


def plotLines(ax, pos, labels):
    for p in pos:
        ax.axvline(x=p, color='k')
    ax.set_xticks(pos, labels)
    ax.set_xlim([pos[0], pos[-1]])
    ax.set_ylabel("E [eV]", labelpad=-5)


def plotPath_QE(ax, fname):
    xPos, energy_eV = np.loadtxt(fname).T
    xPos = (xPos - xPos[0]) / (xPos[-1] - xPos[0])
    ax.plot(xPos, energy_eV, '.', color='gray', label="QE") 




if __name__ == "__main__":







    sc = 0.85
    fig, ax = plt.subplots(1, 2, figsize=(sc*10, sc*4.5))
    axMoS2, axSi = ax
    MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
                 'M' : np.array([0.5, 0, 0]),
                 'K' : np.array([1/3, 1/3, 0]),
                }

    Si_labelToK = { 'G' : np.array([0, 0, 0]),
                 'L' : np.array([0.0, 0.5, 0.0]),
                 'X' : np.array([-0.5, 0, -0.5]),
                 'U' : np.array([-0.375, 0.25, -0.375]),
                 'K' : np.array([0.375, -0.375, 0]),
                }
    argsMLWF = {'label' :'MLWF', 'ls' : '--', 'color' : 'orange', 'ms' : 2 }
    argsCBVB = {'label' :'CB-VB', 'ls' : '-.', 'color' : 'k', 'ms' : 1 }

    plotPath_QE(axMoS2, "../raw_data/mos2_bands.dat.gnu")
    pos = plotPath(axMoS2, "MoS2_10x10x1_tb.npz", MoS2_labelToK, "GMKG", **argsMLWF)
    pos = plotPath(axMoS2, "MoS2_10x10x1_full_tb.npz", MoS2_labelToK, "GMKG", **argsCBVB) 
    plotLines(axMoS2, pos, (r'$\Gamma$', 'M', 'K', r'$\Gamma$'))
    axMoS2.set_ylim([-15, -0.8])
    axMoS2.legend(ncols=3, columnspacing=0.8)
    
    axSi.axhline(y=10, color='dimgray', ls='--')
    plotPath_QE(axSi, "../raw_data/si_bands.dat.gnu")
    pos = plotPath(axSi, "Si_10x10x10_tb.npz", Si_labelToK, "LGXUG", **argsMLWF)
    pos = plotPath(axSi, "Si_10x10x10_full_tb.npz", Si_labelToK, "LGXUG", **argsCBVB)
    plotLines(axSi, pos, ('L', r'$\Gamma$', 'X', 'U', r'$\Gamma$'))
    axSi.set_ylim([-12, 17])
    axSi.legend(ncols=3, columnspacing=0.8)

    place_labels(fig, ax)

    fig.tight_layout()
    save_pdf(fig, "bandstructures")
    # plt.show()

