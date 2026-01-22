from ase.build import mx2
from ase.visualize import view

system = mx2(size=(10, 10, 1))
view(system)
