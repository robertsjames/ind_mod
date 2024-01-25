"""
01/24, R. S. James
"""

import ind_mod as im

import numpy as np
import multihist as mh

export, __all__ = im.exporter()

class RateHistHelper():
    def __init__(self, data=None, 
                 annual_cycles=None, exposures=None,
                 bins_per_cycle=20):
        self.data = data
        self.annual_cycles = annual_cycles
        self.exposures = exposures
        self.bins_per_cycle = bins_per_cycle

    def get_hist(self):
        all_residuals_bins = []
        all_residuals_counts = []
        first_annual_cycle = True

        for annual_cycle in self.annual_cycles.keys():
            data_this_cycle = self.data[self.data['annual_cycle'] == annual_cycle]

            rate_this_cycle = mh.Hist1d(data_this_cycle['time'], bins=self.bins_per_cycle)
            rate_normalisation = self.exposures[annual_cycle]
            rate_this_cycle.histogram = rate_this_cycle.histogram / rate_normalisation

            all_residuals_counts.append(rate_this_cycle.histogram)
            if first_annual_cycle:
                all_residuals_bins.append(rate_this_cycle.bin_edges)
                first_annual_cycle = False
            else:
                all_residuals_bins.append(rate_this_cycle.bin_edges[1:])

        all_residuals_bins = np.concatenate(all_residuals_bins)
        all_residuals_counts = np.concatenate(all_residuals_counts)

        hist = mh.Hist1d.from_histogram(all_residuals_counts, bin_edges=all_residuals_bins)

        return hist

@export
class Analysis():
    def __init__(self, spectrum_file_folder=None, model_file=None):
        self.bg_model = im.BackgroundModel(spectrum_file_folder=spectrum_file_folder, model_file=model_file)

    def get_toy_rate_hist(self, bins_per_cycle=20):
        toy_data = self.bg_model.sample_model()
        annual_cycles = self.bg_model.annual_cycles
        exposures = self.bg_model.exposures

        hist_helper = RateHistHelper(data=toy_data, 
                                     annual_cycles=annual_cycles, exposures=exposures)
        hist = hist_helper.get_hist()

        return hist