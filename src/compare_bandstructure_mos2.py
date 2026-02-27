from src.from_tb.extract_properties_tbfile import MoS2_labelToK, bandstructure_orth_basis
from src.from_tb.utils import plot_bands
import from_tb.parse_and_FT as parse_and_FT
from ase.spectrum.band_structure import BandStructure
import numpy as np
import sys
import matplotlib.pyplot as plt


"""Plot the bandstructure compared to the DFTB+ result"""

# bands_dftb = parse_dftb_band(filepath='dftb_bands/band_graphene.out', n_bands=8)
# bands_dftb = [bands_dftb[:100], bands_dftb[100:200], bands_dftb[200:300]]

"""compared to the QE-result"""
bs = BandStructure.read('./QE_bands/bs_qe_mos2.json')
bands_qe = bs.energies[0]
reference = bs.reference
bands_qe = [bands_qe[:21], bands_qe[21:42], bands_qe[42:63]]
num_seg = [21, 21, 21] 
bs.plot()


def kpts_to_segments(path:str, kpts, num_per_seg, lattice):
    recipLattice = 2*np.pi * np.linalg.inv(lattice).T
    segments = []
    lastRelPos = 0
    labels = [ [path[0], lastRelPos] ]
    idx_count = 0
    for i, num in enumerate(num_per_seg):
        start = kpts[idx_count]
        idx_count += num
        end = kpts[idx_count]
        h = np.linspace(0, 1, num)
        relPos = lastRelPos + h * np.linalg.norm(recipLattice.T @ (end-start))
        kPoints = start + h[:, None] * (end-start)
        segments.append( [kPoints, relPos] )
        # print(f"{s} -> {e} : {kPoints[0]} -> {kPoints[-1]}")
        lastRelPos = relPos[-1]
        labels.append([path[i+1], lastRelPos])
    # normalize pos
    for s in segments:
        s[1] /= lastRelPos
    for l in labels:
        l[1] /= lastRelPos
    return segments, labels


# lattice, cells, degeneracies, Hr, Sr, Rr = w90.read_tb('seedname_mos2_ase.dat')
# segments, labels, bands_alex, H_orth, S_orth, d_orth = bandstructure_orth_basis(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies)
# bands_alex = [w90.au_to_eV(b) for b in bands_alex]

lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb('seedname_tb.dat')
segments, labels, bands_orth, H_orth, S_orth, d_orth = bandstructure_orth_basis(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies)
bands_orth = [parse_and_FT.au_to_eV(b) for b in bands_orth]

bands_orth_shifted = [(b+ reference - bands_orth[0][0,8]) for b in bands_orth]

segments_qe, labels_qe = kpts_to_segments(path='GMKG', kpts=bs.path.kpts, num_per_seg=num_seg, lattice=lattice)

bands = [
         bands_orth_shifted,
         bands_qe
         ]

segment_list = [
    segments,
    segments_qe
]

legend = ['SK',
          'QE']
plot_bands(segments=segment_list, labels=labels, legend=legend, bandstructures=bands, pltname='mos2_bandstructure')