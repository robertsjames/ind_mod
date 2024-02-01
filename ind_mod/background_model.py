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
    def __init__(self, model_file,
                 energy_min=2., energy_max=6.):
        self.energy_min = energy_min
        self.energy_max = energy_max
        
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
            self.background_model[background_component] = im.Spectrum(component=background_component,
                                                                      scale_factor=float(scale_factor),
                                                                      exposure_factor=total_exposure,
                                                                      energy_min=self.energy_min,
                                                                      energy_max=self.energy_max)

        assert self.annual_cycles.keys() == self.exposures.keys(), \
            'Must specify an exposure for each annual cycle'
        
    def get_temporal_info(self, component_name):
        try:
            half_life_years = self.config.getfloat('half_life_years', component_name)
        except Exception:
            raise RuntimeError(f'Could not find half life for component: {component_name}')
        time_constant_ns = (half_life_years/ np.log(2)) * 1e9 * 3600. * 24. * 365.25

        t_start_global = self.annual_cycles[list(self.annual_cycles)[0]][0].value
        t_end_global = self.annual_cycles[list(self.annual_cycles)[-1]][1].value
        t_total = t_end_global - t_start_global
        decay_factor = time_constant_ns / t_total * (1. - np.exp(-t_total / time_constant_ns))

        return t_start_global, time_constant_ns, decay_factor

    def sample_component(self, component_name):
        t_start_global, time_constant_ns, decay_factor = self.get_temporal_info(component_name)

        try:
            df_sample = self.background_model[component_name].sample(decay_factor=decay_factor)
        except Exception:
            raise RuntimeError(f'Component not found in background model: {component_name}')

        weights = []
        for cycle, times in self.annual_cycles.items():
            t_cycle = times[1].value - times[0].value
            cycle_integral = (np.exp(-(times[0].value - t_start_global) / time_constant_ns) - \
                np.exp(-(times[1].value - t_start_global) / time_constant_ns)) / t_cycle
            weights.append(cycle_integral * self.exposures[cycle])
        weights = weights / np.sum(weights)

        cycles_sample = np.random.choice(list(self.annual_cycles.keys()), size=len(df_sample), p=weights)

        times_sample = []
        annual_cycles = []
        for cycle, times in self.annual_cycles.items():
            n_sample_cycle = len(cycles_sample[cycles_sample == cycle])
            b = (times[1].value - times[0].value) / time_constant_ns
            sampled_times = stats.truncexpon.rvs(b, loc=times[0].value,
                                                 scale=time_constant_ns, size=n_sample_cycle)
            times_sample.extend(sampled_times)
            annual_cycles.extend([cycle] * len(sampled_times))

        df_sample['time'] = times_sample
        df_sample['annual_cycle'] = annual_cycles

        return df_sample

    def sample_model(self):
        df_sample_full = pd.DataFrame()
        df_sample_full_list = []

        for background_component in self.background_model.keys():
            df_sample_component = self.sample_component(background_component)
            df_sample_component['source'] = background_component
            df_sample_full_list.append(df_sample_component)

        df_sample_full = pd.concat(df_sample_full_list, ignore_index=True)

        return df_sample_full
        