"""Create Wannier90 style tight binding file in an orthogonal basis"""

from tb_calculations.backtransform_realspace import nonorthogonal_wannierfile, compare_intermediate_steps, orthogonal_wannierfile

orthogonal_wannierfile(filename_in='data/seedname_mos2/seedname_tb.dat', shape_tuple=(5,5,1))
# nonorthogonal_wannierfile(filename_in='./seedname_mos2/seedname_tb.dat', shape_tuple=(10,10,1))

