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
    def __init__(self, spectrum_file_folder=None, root_type=True,
                 component=None, mode='Veto50keV', variable='CrystalEnergySmear0-20',
                 scale_factor=1., exposure_factor=1.):
        assert root_type, 'Currently only support reading spectra from .root files'
        try:
            spectrum = ur.open(f'{spectrum_file_folder}/{component}.root:{mode};1')[f'{variable};1']
        except Exception:
            raise RuntimeError('Error extracting requested information from file')

        self.energy_edges = spectrum.to_numpy()[1]
        spectrum_values = spectrum.to_numpy()[0] * scale_factor * exposure_factor

        self.hist = mh.Histdd.from_histogram(histogram=spectrum_values, bin_edges=[self.energy_edges])

    def get_mu(self, energy_min=None, energy_max=None,
               decay_factor=1.):
        if energy_min is None:
            energy_min = self.energy_edges[0]
        if energy_max is None:
            energy_max = self.energy_edges[-1]

        sliced_hist = self.hist.slice(start=energy_min, stop=energy_max) * decay_factor
        sliced_hist_ebp = sliced_hist * sliced_hist.bin_volumes()
        mu = sliced_hist_ebp.n

        return mu

    def sample(self, energy_min=None, energy_max=None,
               decay_factor=1., for_pdf=False, pdf_stats=int(1e6)):
        if energy_min is None:
            energy_min = self.energy_edges[0]
        if energy_max is None:
            energy_max = self.energy_edges[-1]

        sliced_hist = self.hist.slice(start=energy_min, stop=energy_max) * decay_factor

        if for_pdf is False:
            mu = self.get_mu(energy_min=energy_min, energy_max=energy_max,
                             decay_factor=decay_factor)
            n_sample = np.random.poisson(mu)
        else:
            n_sample = pdf_stats

        energies_sample = sliced_hist.get_random(n_sample)
        df_sample = pd.DataFrame(dict(zip(['energy'], energies_sample.T)))
        
        return df_sample
        