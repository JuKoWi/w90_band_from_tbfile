import w90
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from bandstructure import parsePath,plotLines

# read seedname_tb.dat with read_tb -> realspace matrices for different lattice vectors
# use Hk or Dk to convert realspace to reciprocal space matrices for given k-point 
# get k-points of interest with parsePath
# for each k point calculate r in orthogonal basis by löwdin othogonalization
lattice, cells, degeneracy, Hr, Sr, Rr = w90.read_tb("seedname_tb.dat")
MoS2_labelToK = { 'G' : np.array([0, 0, 0]),
             'M' : np.array([0.5, 0, 0]),
             'K' : np.array([1/3, 1/3, 0]),
            }
segments, labels = parsePath("GMKG", lattice, labelToK=MoS2_labelToK)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
scale = 800

Hk = w90.Hk(cells=cells, H=Hr, kFrac=[0,0,0]) 
Sk = w90.Hk(cells=cells, H=Sr, kFrac=[0,0,0]) 
vals, vecs = sp.linalg.eig(Hk, Sk)
print(vals)

for i, (kPoints, relPos) in enumerate(segments):
    Hk = w90.Hk(cells=cells,H=Hr, kFrac=kPoints)
    Sk = w90.Hk(cells=cells, H=Sr, kFrac=kPoints)
    vals, vecs = sp.linalg.eig(Hk, Sk)
    ax.plot(relPos, vals, '.', color='orange', ms=1)
    print(np.shape(kPoints))
    print(np.shape(relPos))
fig.tight_layout()
l, pos = zip(*labels)
plotLines(ax=ax, pos=pos, labels=labels)
plt.show()



# vals, vr = sp.linalg.eig(Hgamma, Sgamma)
# kFrac = [0,0,0]
# Hgamma = w90.Hk(cells=cells, H=Hr, kFrac=kFrac)
# Sgamma = w90.Hk(cells=cells, H=Sr, kFrac=kFrac)
# Dgamma = w90.Dk(cells=cells, D=Rr, kFrac=kFrac)
