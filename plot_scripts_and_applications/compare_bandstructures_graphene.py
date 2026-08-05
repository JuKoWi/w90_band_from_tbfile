"""Generate plots comparing band structures of graphene from DFTB and the self-generated tight binding files"""

from src.tb_calculations.utils import plot_bands, MoS2_labelToK, parse_dftb_band
from tb_calculations.extract_observables import bandstructure_orth_basis
import tb_calculations.parse_and_FT as parse_and_FT

"""Plot the bandstructure compared to the DFTB+ result"""

bands_dftb = parse_dftb_band(filepath='dftb_bands/band_graphene.out', n_bands=8)
bands_dftb = [bands_dftb[:100], bands_dftb[100:200], bands_dftb[200:300]]

lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb('./seedname_graphene/seedname_tb.dat')
segments, labels, bands_orth, H_orth, S_orth, d_orth = bandstructure_orth_basis(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies)
bands_orth = [parse_and_FT.au_to_eV(b) for b in bands_orth]

bands = [
    # bands_dftb, 
    bands_orth]
legend = [
    # 'DFTB', 
    'SK']
segment_list = [segments, 
                # segments
                ]
plot_bands(segments=segment_list, labels=labels,legend=legend, bandstructures=bands, pltname='graphene_bandstructure')


# legend = ['SK',
#           'QE']
# print(f"bandgap QE: {bands_qe[2][0,13] - bands_qe[2][0,12]}")
# print(f"bandgap SK: {bands_orth_shifted[2][0,9] - bands_orth_shifted[2][0,8]}")