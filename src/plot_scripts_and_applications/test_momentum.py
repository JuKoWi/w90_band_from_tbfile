"""
Comparison of momentum and velocity
"""
from tb_calculations.parse_and_FT import read_tb_momentum, read_tb
from tb_calculations.utils import k_grid, bohr_to_angstrom
from tb_calculations.extract_observables import get_momentum_bloch_realspace, velocity_bloch_lee
import numpy as np

kFrac = k_grid(n_points=(10,10,1))

lattice, cells, degeneracies, Hr, Sr, pr = read_tb_momentum("seedname_tb_momentum.dat") 
p_direct = get_momentum_bloch_realspace(lattice=lattice, cells=cells, degeneracies=degeneracies, Hr=Hr, Sr=Sr, pr=pr, kPoints=kFrac)

lattice, cells, degeneracies, Hr, Sr, Rr = read_tb("seedname_tb.dat") 
p_lee = velocity_bloch_lee(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice_au=lattice)

print(np.allclose(a=p_direct, b=p_lee, atol=1e-4))