from ase.build import mx2
from ase.visualize import view

system = mx2(size=(1, 1, 1))
system.pbc= (True, True, True)
view(system)
