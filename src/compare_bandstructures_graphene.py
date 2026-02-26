from src.from_tb.extract_properties_tbfile import plot_bands, MoS2_labelToK, parse_dftb_band, bandstructure_orth_basis
import from_tb.parse_and_FT as parse_and_FT

"""Plot the bandstructure compared to the DFTB+ result"""

bands_dftb = parse_dftb_band(filepath='dftb_bands/band_graphene.out', n_bands=8)
bands_dftb = [bands_dftb[:100], bands_dftb[100:200], bands_dftb[200:300]]

lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb('seedname_input/seedname_tb_graphene.dat')
segments, labels, bands_orth, H_orth, S_orth, d_orth = bandstructure_orth_basis(lattice=lattice, cells=cells, Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies)
bands_orth = [parse_and_FT.au_to_eV(b) for b in bands_orth]

bands = [bands_dftb, bands_orth]
plot_bands(segments=segments, labels=labels, bandstructures=bands, pltname='graphene_compare_dftb')
