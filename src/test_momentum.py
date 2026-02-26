from from_tb.parse_and_FT import read_tb_momentum, read_tb
from from_tb.utils import k_grid
from from_tb.extract_properties_tbfile import get_momentum_bloch_file, get_momentum_bloch_lee
import numpy as np

kFrac = k_grid(n_points=(10,10,1))

lattice, cells, degeneracies, Hr, Sr, pr = read_tb_momentum("seedname_tb_momentum.dat") 
p_direct = get_momentum_bloch_file(lattice=lattice, cells=cells, degeneracies=degeneracies, Hr=Hr, Sr=Sr, pr=pr, kPoints=kFrac)

lattice, cells, degeneracies, Hr, Sr, Rr = read_tb("seedname_tb.dat") 
p_lee = get_momentum_bloch_lee(Hr=Hr, Sr=Sr, Rr=Rr, degeneracies=degeneracies, kPoints=kFrac, cells=cells, lattice=lattice)

print(np.allclose(a=p_direct, b=p_lee, atol=1e-4))