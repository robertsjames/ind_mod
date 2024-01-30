"""
01/24, R. S. James
"""

import ind_mod as im

import numpy as np
import multihist as mh

export, __all__ = im.exporter()

class RateResidualsHelper():
    def __init__(self, data=None, 
                 annual_cycles=None, exposures=None,
                 bins_per_cycle=10,
                 energy_min=2., energy_max=6.):
        self.data = data
        self.annual_cycles = annual_cycles
        self.exposures = exposures
        self.bins_per_cycle = bins_per_cycle
        self.energy_min = energy_min
        self.energy_max = energy_max

    def get_rate_residuals(self):
        all_time_bins = []
        all_rates = []
        all_residuals = []
        all_errors = []
        
        first_annual_cycle = True

        for annual_cycle in self.annual_cycles.keys():
            data_this_cycle = self.data[self.data['annual_cycle'] == annual_cycle]

            rate_this_cycle = mh.Hist1d(data_this_cycle['time'], bins=self.bins_per_cycle)

            rate_normalisation = (self.exposures[annual_cycle] / self.bins_per_cycle) * \
                (self.energy_max - self.energy_min)
            
            all_rates.append(rate_this_cycle.histogram / rate_normalisation)

            average_rate_this_cycle = np.mean(all_rates[-1])
            all_residuals.append(all_rates[-1] - average_rate_this_cycle)

            poisson_errors = np.sqrt(rate_this_cycle.histogram)  / rate_normalisation
            error_on_mean = np.sqrt(sum(poisson_errors**2)) / len(poisson_errors) * np.ones_like(poisson_errors)
            all_errors.append(np.sqrt(poisson_errors**2 + error_on_mean**2))

            if first_annual_cycle:
                all_time_bins.append(rate_this_cycle.bin_edges)
                first_annual_cycle = False
            else:
                all_time_bins.append(rate_this_cycle.bin_edges[1:])

        all_time_bins = np.concatenate(all_time_bins)
        all_rates = np.concatenate(all_rates)
        all_residuals = np.concatenate(all_residuals)
        all_errors = np.concatenate(all_errors)

        rate_hist = mh.Hist1d.from_histogram(all_rates, bin_edges=all_time_bins)

        return rate_hist, all_residuals, all_errors

@export
class Analysis():
    def __init__(self, spectrum_file_folder=None, model_file=None,):
        self.bg_model = im.BackgroundModel(spectrum_file_folder=spectrum_file_folder, model_file=model_file)

    def get_toy_rate_residuals(self, bins_per_cycle=20):
        toy_data = self.bg_model.sample_model()
        annual_cycles = self.bg_model.annual_cycles
        exposures = self.bg_model.exposures

        hist_helper = RateResidualsHelper(data=toy_data,
                                          annual_cycles=annual_cycles,
                                          exposures=exposures,
                                          energy_min=self.bg_model.energy_min,
                                          energy_max=self.bg_model.energy_max)
        rate_hist, residuals, errors = hist_helper.get_rate_residuals()

        return rate_hist, residuals, errors
