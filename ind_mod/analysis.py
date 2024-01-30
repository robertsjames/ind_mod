"""
01/24, R. S. James
"""

import ind_mod as im

import numpy as np
import multihist as mh

from scipy.optimize import minimize

export, __all__ = im.exporter()

class RateResidualsHelper():
    def __init__(self, data,
                 annual_cycles, exposures,
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

            times_months = data_this_cycle['time'].values / 1e9 / 3600. / 24. / (365.25 / 12.)
            rate_this_cycle = mh.Hist1d(times_months, bins=self.bins_per_cycle)

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

        all_time_bins = np.concatenate(all_time_bins) - np.concatenate(all_time_bins)[0]
        all_rates = np.concatenate(all_rates)
        all_residuals = np.concatenate(all_residuals)
        all_errors = np.concatenate(all_errors)

        rate_hist = mh.Hist1d.from_histogram(all_rates, bin_edges=all_time_bins)

        return rate_hist, all_residuals, all_errors

class ChisqMinimization1D:
    def __init__(self, bin_centers, bin_values, errors,
                 model_fn):
        self.bin_centers = bin_centers
        self.bin_values = bin_values
        self.errors = errors
        self.model_fn = model_fn

    def get_chisq(self, args):
        model = self.model_fn(self.bin_centers, *args)
        chi_sq = np.sum((self.bin_values - model)**2 / self.errors**2)
        return chi_sq

    def minimise_chisq(self, guess):
        return minimize(
            fun=self.get_chisq,
            x0=guess,
            method='Nelder-Mead')

@export
class Analysis():
    def __init__(self, spectrum_file_folder, model_file,
                 energy_min=2., energy_max=6.):
        self.bg_model = im.BackgroundModel(spectrum_file_folder=spectrum_file_folder, model_file=model_file,
                                           energy_min=energy_min, energy_max=energy_max)

    def get_toy_rate_residuals(self, bins_per_cycle=10):
        toy_data = self.bg_model.sample_model()
        annual_cycles = self.bg_model.annual_cycles
        exposures = self.bg_model.exposures

        hist_helper = RateResidualsHelper(data=toy_data,
                                          annual_cycles=annual_cycles,
                                          exposures=exposures,
                                          bins_per_cycle=bins_per_cycle)
        rate_hist, residuals, errors = hist_helper.get_rate_residuals()

        return rate_hist, residuals, errors

    def do_chisq_fit(self, rate_hist, residuals, errors,
                     model_fn, guess):
        time_bin_edges = rate_hist.bin_edges
        time_bin_centers = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])

        chisq_min_1d = ChisqMinimization1D(time_bin_centers, residuals, errors, model_fn)

        fit = chisq_min_1d.minimise_chisq(guess=guess).x

        return fit
