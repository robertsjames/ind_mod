"""
01/24, R. S. James
"""

import ind_mod as im

import numpy as np
import multihist as mh
import pandas as pd

from scipy.optimize import minimize

export, __all__ = im.exporter()

ns_to_months = 1. / 1e9 / 3600. / 24. / (365.25 / 12.)

class RatesResidualsHelper():
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

    def get_cycle_rates_average_errors(self, annual_cycle):

        cycle_edges = (self.annual_cycles[annual_cycle][0].value * ns_to_months,
                       self.annual_cycles[annual_cycle][1].value * ns_to_months)

        data_this_cycle = self.data[self.data['annual_cycle'] == annual_cycle]

        times = data_this_cycle['time'].values * ns_to_months

        rates_this_cycle = mh.Hist1d(times, bins=self.bins_per_cycle)
        time_bins = rates_this_cycle.bin_edges

        rate_normalisation = (self.exposures[annual_cycle] / self.bins_per_cycle) * \
            (self.energy_max - self.energy_min)

        poisson_errors = np.sqrt(rates_this_cycle.histogram)  / rate_normalisation
        error_on_mean = np.sqrt(sum(poisson_errors**2)) / len(poisson_errors) * np.ones_like(poisson_errors)
        errors = np.sqrt(poisson_errors**2 + error_on_mean**2)

        rates = rates_this_cycle.histogram / rate_normalisation
        average_rate = np.mean(rates)

        return cycle_edges, time_bins, rates, average_rate, errors

    def get_rates_and_residuals(self):
        all_time_bins = []
        all_rates = []
        all_residuals = []
        all_errors = []
        
        first_annual_cycle = True

        for annual_cycle in self.annual_cycles.keys():
            _, time_bins, rates, average_rate, errors = self.get_cycle_rates_average_errors(annual_cycle)

            all_rates.append(rates)
            all_residuals.append(rates - average_rate)
            all_errors.append(errors)

            if first_annual_cycle:
                all_time_bins.append(time_bins)
                first_annual_cycle = False
            else:
                all_time_bins.append(time_bins[1:])

        all_time_bins = np.concatenate(all_time_bins) - np.concatenate(all_time_bins)[0]
        all_rates = np.concatenate(all_rates)
        all_residuals = np.concatenate(all_residuals)
        all_errors = np.concatenate(all_errors)

        rate_hist = mh.Hist1d.from_histogram(all_rates, bin_edges=all_time_bins)

        return rate_hist, all_residuals, all_errors

    def get_cycles_and_average_rates(self):
        all_cycle_endpoints = []
        all_average_rates = []

        first_annual_cycle = True

        for annual_cycle in self.annual_cycles.keys():
            cycle_edges, _, _, average_rate, _ = self.get_cycle_rates_average_errors(annual_cycle)

            all_average_rates.append(average_rate)

            if first_annual_cycle:
                all_cycle_endpoints.extend(list(cycle_edges))
                first_annual_cycle = False
            else:
                all_cycle_endpoints.append(cycle_edges[-1])

        all_cycle_endpoints = np.array(all_cycle_endpoints) - np.array(all_cycle_endpoints)[0]
        all_average_rates = np.array(all_average_rates)

        return all_cycle_endpoints, all_average_rates

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
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.bg_model = im.BackgroundModel(spectrum_file_folder=spectrum_file_folder, model_file=model_file,
                                           energy_min=self.energy_min, energy_max=self.energy_max)

    def get_toy_data(self):
        toy_data = self.bg_model.sample_model()

        return toy_data

    def get_rates_and_residuals(self, toy_data, bins_per_cycle=10):
        annual_cycles = self.bg_model.annual_cycles
        exposures = self.bg_model.exposures

        hist_helper = RatesResidualsHelper(data=toy_data,
                                           annual_cycles=annual_cycles,
                                           exposures=exposures,
                                           bins_per_cycle=bins_per_cycle,
                                           energy_min=self.energy_min, energy_max=self.energy_max)
        rate_hist, residuals, errors = hist_helper.get_rates_and_residuals()
        time_bin_edges = rate_hist.bin_edges
        time_bin_centers = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])

        return rate_hist, time_bin_centers, residuals, errors

    def get_cycles_and_average_rates(self, toy_data, bins_per_cycle=10):
        annual_cycles = self.bg_model.annual_cycles
        exposures = self.bg_model.exposures

        hist_helper = RatesResidualsHelper(data=toy_data,
                                           annual_cycles=annual_cycles,
                                           exposures=exposures,
                                           bins_per_cycle=bins_per_cycle,
                                           energy_min=self.energy_min, energy_max=self.energy_max)
        cycle_endpoints, average_rates = hist_helper.get_cycles_and_average_rates()

        return cycle_endpoints, average_rates

    def get_phase_offset(self, start_month='06', start_day='02'):
        t_start_global = self.bg_model.annual_cycles[list(self.bg_model.annual_cycles)[0]][0]
        t_start_global_year = t_start_global.year

        offset_from = pd.to_datetime(f'{t_start_global_year}-{start_month}-{start_day}T00:00:00')

        phase_offset = (t_start_global.value - offset_from.value) * ns_to_months

        return phase_offset

    def do_chisq_fit(self, time_bin_centers, residuals, errors,
                     model_fn, guess):
        chisq_min_1d = ChisqMinimization1D(time_bin_centers, residuals, errors, model_fn)

        fit = chisq_min_1d.minimise_chisq(guess=guess).x

        return fit
