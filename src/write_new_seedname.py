from from_tb.backtransform_realspace import nonorthogonal_wannierfile, compare_intermediate_steps

file = 'seedname_mos2_ase.dat'
nonorthogonal_wannierfile(filename_in=file, shape_tuple=(20,20,1))
compare_intermediate_steps(filename_A=file, filename_B='seedname_nonorth.dat', shape_tuple=(20,20,1))