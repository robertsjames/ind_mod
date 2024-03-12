"""
01/24, R. S. James
"""

import ind_mod as im

import sys, os

import numpy as np
import pandas as pd
import uproot as ur
import multihist as mh

from scipy import stats

export, __all__ = im.exporter()

@export
class Spectrum():
    def __init__(self, component,
                 root_type=True, mode='Veto50keV', variable='CrystalEnergy0-20',
                 scale_factor=1., exposure_factor=1.,
                 energy_min=2., energy_max=6.):
        self.energy_min = energy_min
        self.energy_max = energy_max
        
        assert root_type, 'Currently only support reading spectra from .root files'
        try:
            spectra_folder = os.path.join(os.path.dirname(__file__), f'data/background_sim_spectra')
            spectrum = ur.open(f'{spectra_folder}/{component}.root:{mode};1')[f'{variable};1']
        except Exception:
            raise RuntimeError('Error extracting requested information from file')

        self.energy_edges = spectrum.to_numpy()[1]
        spectrum_values = spectrum.to_numpy()[0] * scale_factor * exposure_factor

        self.hist = mh.Histdd.from_histogram(histogram=spectrum_values, bin_edges=[self.energy_edges])

    def get_mu(self, decay_factor=1.):
        sliced_hist = self.hist.slice(start=(0.5 * self.energy_min), stop=(2. * self.energy_max)) * decay_factor
        sliced_hist_ebp = sliced_hist * sliced_hist.bin_volumes()
        mu = sliced_hist_ebp.n

        return mu

    def sample(self, decay_factor=1.):
        sliced_hist = self.hist.slice(start=(0.5 * self.energy_min), stop=(2. * self.energy_max)) * decay_factor

        mu = self.get_mu(decay_factor=decay_factor)
        n_sample = np.random.poisson(mu)

        energies_sample = sliced_hist.get_random(n_sample)[:, 0]

        sigma_energies_sample = 0.488 * np.sqrt(energies_sample) + 0.0091 * energies_sample
        energies_sample_smear = np.array([stats.norm.rvs(loc=energies_sample, scale=sigma_energies_sample)])

        energies_sample_smear = energies_sample_smear[np.where(energies_sample_smear >= self.energy_min)]
        energies_sample_smear = energies_sample_smear[np.where(energies_sample_smear <= self.energy_max)]

        df_sample = pd.DataFrame(dict(zip(['energy'], [energies_sample_smear])))
        
        return df_sample
        