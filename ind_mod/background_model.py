"""
01/24, R. S. James
"""

import ind_mod as im

import sys, os
import configparser

import numpy as np
import pandas as pd

from scipy import stats

export, __all__ = im.exporter()

@export
class BackgroundModel():
    def __init__(self, spectrum_file_folder=None, model_file=None):
        try:
            self.config = configparser.ConfigParser(inline_comment_prefixes=';')
            self.config.optionxform = str
            self.config.read(os.path.join(os.path.dirname(__file__), f'models/{model_file}.ini'))
        except Exception:
            raise RuntimeError(f'Could not find specified model file: {model_file}')

        self.annual_cycles = dict()
        for (cycle, times) in self.config.items('annual_cycles'):
            times = times.split()
            assert len (times) == 2, 'Cycle can only be composed of two times (start and end)'
            t_start = pd.to_datetime(times[0])
            t_end = pd.to_datetime(times[1])
            self.annual_cycles[cycle] = (t_start, t_end)

        self.exposures = dict()
        for (cycle, exposure) in self.config.items('exposure_kgday'):
            self.exposures[cycle] = float(exposure)

        total_exposure = sum(list(self.exposures.values()))
        self.background_model = dict()
        for (background_component, scale_factor) in self.config.items('component_scalings'):
            self.background_model[background_component] = im.Spectrum(spectrum_file_folder=spectrum_file_folder,
                                                                      component=background_component,
                                                                      scale_factor=float(scale_factor),
                                                                      exposure_factor=total_exposure)

        assert self.annual_cycles.keys() == self.exposures.keys(), \
            'Must specify an exposure for each annual cycle'

    def sample_component(self, component_name):
        try:
            half_life_years = self.config.getfloat('half_life_years', component_name)
        except Exception:
            raise RuntimeError(f'Could not find half life for component: {component_name}')
        time_constant_ns = (half_life_years/ np.log(2)) * 1e9 * 3600. * 24. * 365.25

        weights = []
        for cycle, times in self.annual_cycles.items():
            cycle_integral = np.exp(-times[0].value / time_constant_ns) - np.exp(-times[1].value / time_constant_ns)
            weights.append(cycle_integral * self.exposures[cycle])
        weights = weights / np.sum(weights)

        try:
            df_sample = self.background_model[component_name].sample()
        except Exception:
            raise RuntimeError(f'Component not found in background model: {component_name}')

        cycles_sample = np.random.choice(list(self.annual_cycles.keys()), size=len(df_sample), p=weights)

        times_sample = []
        for cycle, times in self.annual_cycles.items():
            n_sample_cycle = len(cycles_sample[cycles_sample == cycle])
            b = (times[1].value - times[0].value) / time_constant_ns
            sampled_times = stats.truncexpon.rvs(b, loc=times[0].value,
                                                 scale=time_constant_ns, size=n_sample_cycle)
            times_sample.extend(sampled_times)

        df_sample['time'] = times_sample

        return df_sample
        