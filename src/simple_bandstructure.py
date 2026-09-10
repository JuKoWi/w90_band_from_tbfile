from tb_calculations.bandstructure import Bands, plot_bandstructures
from tb_calculations import utils

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
bs = Bands.from_custom_tb(tb_file='data/seedname_mos2/seedname_tb.dat', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
bs.ef_to_zero(Ef=-4.6)
# bs_dftb.ef_to_zero(Ef=-4.6)

"""Graphene"""
# bs = Bands.from_custom_tb(tb_file='data/seedname_graphene/seedname_tb.dat', bz_path='GMKG', label_to_k=utils.MOS2_LABEL_TO_K)
# bs_dftb = Bands.from_dftbplus(outfile='band.out',
#                                  bz_path='GMKG',
#                                        label_to_k=utils.MOS2_LABEL_TO_K,
#                                          n_bands=8,
#                                          points_per_segment=[101, 100, 100],
#                                          lattice_ang=np.array([[2.46, 0.0, 0.0], [-1.23, 2.130422493309719, 0.0], [0.0, -0.0, 20.0]]),
#                                          )

"""Diamond"""
# bs =Bands.from_custom_tb(tb_file='data/seedname_diamond/seedname_tb.dat', bz_path='LGXL', label_to_k=utils.FCC_LABEL_TO_K)

plot_bandstructures([bs], 
                    # energy_range_eV=(-10, 5)
                    )
