"""
01/24, R. S. James
"""

import ind_mod as im

import numpy as np
import pandas as pd
import uproot as ur
import multihist as mh

export, __all__ = im.exporter()

@export
class Spectrum():
    def __init__(self, root_path=None, 
                 mode='Veto50keV', variable='CrystalEnergySmear0-20'):
        try:
            spectrum = ur.open(f'{root_path}:{mode};1')[f'{variable};1']
        except Exception:
            raise RuntimeError("Could not find specified variable for specified mode in specified file")

        self.energy_edges = spectrum.to_numpy()[1]
        spectrum_values = spectrum.to_numpy()[0]

        self.hist = mh.Histdd.from_histogram(histogram=spectrum_values, bin_edges=[self.energy_edges])

        self.hist.bin_edges
        self.hist.histogram

    def sample(self, energy_min=None, energy_max=None):
        if energy_min is None:
            energy_min = self.energy_edges[0]
        if energy_max is None:
            energy_max = self.energy_edges[-1]
        
        sliced_hist = self.hist.slice(start=energy_min, stop=energy_max)

        sliced_hist_ebp = sliced_hist * sliced_hist.bin_volumes()
        mu = sliced_hist_ebp.n
        n_sample = np.random.poisson(mu)

        data_sample = sliced_hist.get_random(n_sample)
        df_sample = pd.DataFrame(dict(zip(['energy'], data_sample.T)))
        
        return df_sample
        