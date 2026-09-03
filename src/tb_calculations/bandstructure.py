from tb_calculations import parse_and_FT
from tb_calculations import extract_observables
from tb_calculations import utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
plt.rcParams.update({'font.size': 16})
plt.rcParams['savefig.bbox'] = 'tight'

class Bands():
    """
    Contains bandstructure information to show in a plot:
        - symmetry point labels
        - relative positions along the whole path
        - number of points per segment including the symmetry point at the beginning 
        - information about the method
        - energy values 
        - E_f
        - number of bands
    """

    def __init__(self):
        self.method = None
        self.path = None
        self.relpos_segments = None
        self.bands_eV = None
        self.e_fermi = None

    @classmethod
    def from_custom_tb(cls, tb_file, bz_path, label_to_k):
        self = cls()
        self.method = "Custom TB"
        self.path = bz_path
        lattice, cells, degeneracies, Hr, Sr, Rr = parse_and_FT.read_tb(tb_file)
        segments, labels = parse_and_FT.parsePath(path=bz_path, lattice=lattice, label_to_k=label_to_k)
        self.relpos_segments = [s[1] for s in segments]

        ksegments = [s[0] for s in segments]
        bands = extract_observables.bandstructure_from_k(tb_file=tb_file, ksegments=ksegments)
        self.bands_eV = [parse_and_FT.au_to_eV(b) for b in bands]
        self._normalize_relpos
        return self

    @classmethod
    def from_dftbplus(cls, outfile, n_bands, bz_path, label_to_k, points_per_segment, lattice_ang):
        self = cls()
        self.method = "DFTB+"
        self.path = bz_path
        self.bands_eV = utils.parse_dftb_band(filepath=outfile, n_bands=n_bands, points_per_segment=points_per_segment)
        self.relpos_segments = self._kpoints_to_relpos(lattice_ang=lattice_ang, bz_path=bz_path, label_to_k=label_to_k, points_per_segment=points_per_segment)
        self._normalize_relpos()
        return self

    def _kpoints_to_relpos(self, lattice_ang, bz_path, label_to_k, points_per_segment):
        lattice_bohr = utils.angstrom_to_bohr(lattice_ang)
        rec_lattice = 2*np.pi * np.linalg.inv(lattice_bohr).T
        relpos_segments = []
        last_rel_pos = 0
        for i, (s, e) in enumerate(zip(bz_path[:-1], bz_path[1:])):
            if not s.isalpha() or not e.isalpha():
                continue
            start_k = label_to_k[s]
            end_k = label_to_k[e]
            h = np.linspace(0, 1, points_per_segment[i])
            if self.method == "DFTB+":
                h = np.linspace(0,1, points_per_segment[i], endpoint=False)
                h += h[1]
            if self.method == "DFTB+" and (i == 0): 
                h = np.linspace(0, 1, points_per_segment[i])
            relPos = last_rel_pos + h * np.linalg.norm(rec_lattice.T @ (end_k-start_k))
            relpos_segments.append(relPos)
            last_rel_pos = relPos[-1]
        return relpos_segments

    def ef_to_zero(self, Ef):
        for i, b in enumerate(self.bands_eV):
            self.bands_eV[i] = b - Ef
        self.efermi = 0

    def _normalize_relpos(self):
        max = self.relpos_segments[-1][-1]
        for i, s in enumerate(self.relpos_segments):
            self.relpos_segments[i] = s/max
            

def plot_bandstructures(bs_list, energy_range_eV=None):
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    scale = 800
    colors = mcolors.TABLEAU_COLORS
    names = list(colors)
    for i, bs in enumerate(bs_list):
        ax.plot([], [], color=colors[names[i]], label=bs.method)
        for j, r in enumerate(bs.relpos_segments):
            plt.plot(r, bs.bands_eV[j], '.', color=colors[names[i]], ms=2) 

    bs = bs_list[0]
    pos_lines = []
    for seg in bs.relpos_segments:
        pos_lines.append(seg[0])
    pos_lines.append(bs.relpos_segments[-1][-1])
    assert len(pos_lines) == len(bs.path)
    utils.plotLines(ax=ax, pos=pos_lines, labels=bs.path)
    if energy_range_eV is not None:
        ax.set_ylim(bottom=energy_range_eV[0], top=energy_range_eV[1])
    ax.legend()
    plt.show()

