from tb_calculations.bandstructure import Bands, plot_bandstructures, colorcoded_bandstructures
from tb_calculations.parse_and_FT import parse_hdf5
from tb_calculations import utils
import numpy as np

# bandstructure.bandstructure_plot('data/seedname_diamond/seedname_tb.dat', 'GXWKGLUWLK', utils.FCC_LABEL_TO_K)
# bandstructure.bandstructure_plot('data/seedname_gold/seedname_tb.dat', 'GXWKGLWUX', utils.FCC_LABEL_TO_K)
# bandstructure.bandstructure_plot('data/seedname_aluminium/seedname_tb.dat', 'LGXUG', utils.FCC_LABEL_TO_K)
# bandstructure.bandstructure_plot('data/seedname_aluminium/seedname_tb.dat', 'WXGL', utils.FCC_LABEL_TO_K)
# bandstructure.bandstructure_plot('data/seedname_GaAs/seedname_tb.dat', 'LGXK', utils.FCC_LABEL_TO_K)

"""MoS2"""
# bs_dftb = Bands.from_dftbplus(outfile='band.out',
#                                 bz_path='GMKG',
#                                       label_to_k=utils.MOS2_LABEL_TO_K,
#                                         n_bands=27,
#                                         points_per_segment=[101, 100, 100],
#                                         lattice_ang=np.array([[3.18, 0.0, 0.0], [-1.59, 2.753960784034515, 0.0], [0.0, -0.0, 43.19]]),
#                                         )
# bs_qe = Bands.from_custom_tb(tb_file='./data/martin_qe_mos2_ref/MoS2_custom_tb.dat', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K, orthogonal=True)
# bs2 = Bands.from_seedname(tb_file='data/seedname_mos2/seedname_tb.dat', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# bs2 = Bands.from_hdf5(tb_file='mos2_sk.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='mos2_sk.hdf5')
# bs2.ef_to_zero(Ef=-4.6)

# bs3 = Bands.from_hdf5(tb_file='modified_mos2.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='modified_mos2.hdf5')
# bs3.ef_to_zero(Ef=-4.6)
# bs3.set_momentum(tb_model=tb_model)
# bs3.set_velocity(tb_model=tb_model)

"""Graphene"""
# bs = Bands.from_custom_tb(tb_file='data/seedname_graphene/seedname_tb.dat', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# bs_dftb = Bands.from_dftbplus(infile='band.out',
#                                  bz_path='GMKG',
#                                        label_to_k=utils.MOS2_LABEL_TO_K,
#                                          n_bands=8,
#                                          points_per_segment=[101, 100, 100],
#                                          lattice_ang=np.array([[2.46, 0.0, 0.0], [-1.23, 2.130422493309719, 0.0], [0.0, -0.0, 20.0]]),
#                                          )
# bs = Bands.from_hdf5(tb_file='graphene_sk.hdf5', bz_path="GMKG", label_to_k=utils.MOS2_LABEL_TO_K)
# bs2 = Bands.from_hdf5(tb_file='modified_graphene.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='graphene_sk.hdf5')
# bs.set_momentum(tb_model=tb_model)
# bs.set_velocity(tb_model=tb_model)
# bs.ef_to_zero(Ef=-4.453)


"""Diamond"""
# bs =Bands.from_custom_tb(tb_file='data/seedname_diamond/seedname_tb.dat', bz_path='LGXL', label_to_k=utils.FCC_LABEL_TO_K)

"""GaAs"""
# bs1 = Bands.from_hdf5(tb_file='GaAs_sk.hdf5', bz_path='LGXK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel1 = parse_hdf5(filename="GaAs_sk.hdf5")
# bs1.set_velocity(tb_model=tbmodel1)
# bs2 = Bands.from_hdf5(tb_file='GaAs_onsite_expansion.hdf5', bz_path='LGXK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel2 = parse_hdf5(filename="GaAs_onsite_expansion.hdf5")
# bs2.set_velocity(tb_model=tbmodel2)
# bs3 = Bands.from_hdf5(tb_file='GaAs_mulliken.hdf5', bz_path='LGXK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel3 = parse_hdf5(filename='GaAs_mulliken.hdf5')
# bs3.set_velocity(tb_model=tbmodel3)
# bs2 = Bands.from_seedname(tb_file='data/seedname_GaAs/seedname_tb.dat', bz_path='LGXK', label_to_k=utils.FCC_LABEL_TO_K)

"""ZnO wurtzite"""
bs1 = Bands.from_hdf5(tb_file='ZnO_wurtzite_sk.hdf5', bz_path='GMKGALHA', label_to_k=utils.WURTZITE_LABEL_TO_K)
tbmodel1 = parse_hdf5('ZnO_wurtzite_sk.hdf5')
bs1.set_velocity(tb_model=tbmodel1)
bs2 = Bands.from_hdf5(tb_file='ZnO_wurtzite_onsite_expansion.hdf5', bz_path='GMKGALHA', label_to_k=utils.WURTZITE_LABEL_TO_K)
tbmodel2 = parse_hdf5('ZnO_wurtzite_onsite_expansion.hdf5')
bs2.set_velocity(tb_model=tbmodel2)
bs3 = Bands.from_hdf5(tb_file='ZnO_wurtzite_mulliken.hdf5', bz_path='GMKGALHA', label_to_k=utils.WURTZITE_LABEL_TO_K)
tbmodel3 = parse_hdf5('ZnO_wurtzite_mulliken.hdf5')
bs3.set_velocity(tb_model=tbmodel3)

"""ZnO zincblende"""
# bs1 = Bands.from_hdf5(tb_file='ZnO_zincblende_sk.hdf5', bz_path='GXWKGLUWLK', label_to_k=utils.FCC_LABEL_TO_K)
# bs1.ef_to_zero(Ef=-6.34)
# bs2 = Bands.from_hdf5(tb_file='modified_zincblende.hdf5', bz_path='GXWKGLUWLK', label_to_k=utils.FCC_LABEL_TO_K)
# bs2.ef_to_zero(Ef=-2.67)
# bs_dftb = 
# bs1 = Bands.from_hdf5(tb_file='ZnO_zincblende_sk.hdf5', bz_path='GXWKGLUWLK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel1 = parse_hdf5(filename="ZnO_zincblende_sk.hdf5")
# bs1.set_velocity(tb_model=tbmodel1)
# bs2 = Bands.from_hdf5(tb_file='ZnO_zincblende_onsite_expansion.hdf5', bz_path='GXWKGLUWLK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel2 = parse_hdf5(filename="ZnO_zincblende_onsite_expansion.hdf5")
# bs2.set_velocity(tb_model=tbmodel2)
# bs3 = Bands.from_hdf5(tb_file='ZnO_zincblende_mulliken.hdf5', bz_path='GXWKGLUWLK', label_to_k=utils.FCC_LABEL_TO_K)
# tbmodel3 = parse_hdf5(filename='ZnO_zincblende_mulliken.hdf5')
# bs3.set_velocity(tb_model=tbmodel3)

"""hBN"""
# bs1 = Bands.from_hdf5(tb_file='hBN_sk.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='hBN_sk.hdf5')
# bs1.set_velocity(tb_model=tb_model)
# bs2 = Bands.from_hdf5(tb_file='hBN_onsite_expansion.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='hBN_onsite_expansion.hdf5')
# bs2.set_velocity(tb_model=tb_model)
# bs3 = Bands.from_hdf5(tb_file='hBN_mulliken.hdf5', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# tb_model = parse_hdf5(filename='hBN_mulliken.hdf5')
# bs3.set_velocity(tb_model=tb_model)

if True:
# colorcoded_bandstructures(bs=bs3, component='xy', n_occ=9)
# colorcoded_bandstructures(bs=bs3, component='xy', n_occ=9, momentum=True)
    colorcoded_bandstructures(bs=bs1, component='xy', n_occ=18)
    colorcoded_bandstructures(bs=bs2, component='xy', n_occ=18)
    colorcoded_bandstructures(bs=bs3, component='xy', n_occ=18)
# colorcoded_bandstructures(bs=bs, component='xy', n_occ=4, momentum=True)

if True:
    plot_bandstructures([
                        bs1, 
                        #  bs2,
                        #  bs3
                        # bs_dftb
                         ], 
                        energy_range_eV=(-10, 10)
                        )
