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
ns_to_days = 1. / 1e9 / 3600. / 24.

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

    def rate_calc_for_cycle(self, annual_cycle):

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
            _, time_bins, rates, average_rate, errors = self.rate_calc_for_cycle(annual_cycle)

            all_rates.extend(rates)
            all_residuals.extend(rates - average_rate)
            all_errors.extend(errors)

            if first_annual_cycle:
                all_time_bins.extend(time_bins)
                first_annual_cycle = False
            else:
                all_time_bins.extend(time_bins[1:])

        all_time_bins = all_time_bins - all_time_bins[0]

        rate_hist = mh.Hist1d.from_histogram(all_rates, bin_edges=all_time_bins)

        return rate_hist, all_residuals, all_errors

    def get_cycles_and_average_rates(self):
        all_cycle_endpoints = []
        all_average_rates = []

        first_annual_cycle = True
        for annual_cycle in self.annual_cycles.keys():
            cycle_edges, _, _, average_rate, _ = self.rate_calc_for_cycle(annual_cycle)

            all_average_rates.append(average_rate)

            if first_annual_cycle:
                all_cycle_endpoints.extend(list(cycle_edges))
                first_annual_cycle = False
            else:
                all_cycle_endpoints.append(cycle_edges[-1])

        all_cycle_endpoints = np.array(all_cycle_endpoints) - np.array(all_cycle_endpoints)[0]
        all_average_rates = np.array(all_average_rates)

        return all_cycle_endpoints, all_average_rates


class EnergyBinsHelper():
    def __init__(self, data,
                 annual_cycles, exposures,
                 time_bins_per_cycle=10,
                 energy_min=2., energy_max=6.,
                 energy_bin_width=0.5
                 ):
        self.data = data
        self.annual_cycles = annual_cycles
        self.exposures = exposures
        self.time_bins_per_cycle = time_bins_per_cycle
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_bin_width = energy_bin_width

    def get_bins_and_scalings(self):
        energy_bin_edges = np.arange(self.energy_min, self.energy_max + self.energy_bin_width,
                                     self.energy_bin_width)
        time_bin_edges = []
        scalings = []

        first_annual_cycle = True
        for annual_cycle in self.annual_cycles.keys():
            cycle_bins = np.linspace(self.annual_cycles[annual_cycle][0].value * ns_to_days,
                                     self.annual_cycles[annual_cycle][1].value * ns_to_days,
                                     self.time_bins_per_cycle + 1)

            if first_annual_cycle:
                time_bin_edges.extend(cycle_bins)
                first_annual_cycle = False
            else:
                time_bin_edges.extend(cycle_bins[1:])

            scalings.extend([(self.exposures[annual_cycle] / self.time_bins_per_cycle)] * self.time_bins_per_cycle)

        time_bin_edges = time_bin_edges - time_bin_edges[0]

        scalings = np.tile(scalings, [len(energy_bin_edges) - 1, 1])

        return energy_bin_edges, time_bin_edges, scalings

    def get_energy_time_hist(self):
        energy_bin_edges, time_bin_edges, _ = self.get_bins_and_scalings()
        data_energies = self.data['energy'].values
        t_start_global = self.annual_cycles[list(self.annual_cycles)[0]][0].value
        data_times = (self.data['time'].values - t_start_global) * ns_to_days

        energy_time_hist = mh.Histdd(data_energies, data_times,
                                     bins=[energy_bin_edges, time_bin_edges])

        return energy_time_hist


class ChisqMinimization1D:
    def __init__(self, bin_centers, bin_values, errors,
                 model_fn):
        self.bin_centers = bin_centers
        self.bin_values = bin_values
        self.errors = errors
        self.model_fn = model_fn

    def chisq(self, args):
        model = self.model_fn(self.bin_centers, *args)
        chi_sq = np.sum(np.array(self.bin_values - model)**2 / np.array(self.errors)**2)
        return chi_sq

    def minimise_chisq(self, guess):
        return minimize(
            fun=self.chisq,
            x0=guess,
            method='Nelder-Mead')


class BinnedPoissonML:
    def __init__(self, bin_values, scalings):
        self.bin_values = bin_values
        self.scalings = scalings

    def likelihood(self):
        pass


@export
class Analysis():
    def __init__(self, model_file,
                 energy_min=2., energy_max=6.):
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.bg_model = im.BackgroundModel(model_file=model_file,
                                           energy_min=self.energy_min, energy_max=self.energy_max)


    def get_toy_data(self):
        toy_data = self.bg_model.sample_model()

        return toy_data


    def get_rates_and_residuals(self, toy_data, bins_per_cycle=10):
        hist_helper = RatesResidualsHelper(data=toy_data,
                                           annual_cycles=self.bg_model.annual_cycles,
                                           exposures=self.bg_model.exposures,
                                           bins_per_cycle=bins_per_cycle,
                                           energy_min=self.energy_min, energy_max=self.energy_max)
        rate_hist, residuals, errors = hist_helper.get_rates_and_residuals()
        time_bin_edges = rate_hist.bin_edges
        time_bin_centers = 0.5 * (time_bin_edges[1:] + time_bin_edges[:-1])

        return rate_hist, time_bin_centers, residuals, errors

    def get_cycles_and_average_rates(self, toy_data, bins_per_cycle=10):
        hist_helper = RatesResidualsHelper(data=toy_data,
                                           annual_cycles=self.bg_model.annual_cycles,
                                           exposures=self.bg_model.exposures,
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


    def get_energy_time_hist(self, toy_data,
                             time_bins_per_cycle=10, energy_bin_width=0.5):
        hist_helper = EnergyBinsHelper(data=toy_data,
                                       annual_cycles=self.bg_model.annual_cycles,
                                       exposures=self.bg_model.exposures,
                                       time_bins_per_cycle=time_bins_per_cycle,
                                       energy_min=self.energy_min, energy_max=self.energy_max,
                                       energy_bin_width=energy_bin_width)
        energy_time_hist = hist_helper.get_energy_time_hist()

        return energy_time_hist


    def do_chisq_fit(self, time_bin_centers, residuals, errors,
                     model_fn, guess):
        chisq_min_1d = ChisqMinimization1D(time_bin_centers, residuals, errors, model_fn)

        fit = chisq_min_1d.minimise_chisq(guess=guess).x

        return fit
