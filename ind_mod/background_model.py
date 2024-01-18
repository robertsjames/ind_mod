"""
01/24, R. S. James
"""

import ind_mod as im

import sys, os
import configparser

import pandas as pd

export, __all__ = im.exporter()

@export
class BackgroundModel():
    def __init__(self, spectrum_file_folder=None, model_file=None):
        try:
            config = configparser.ConfigParser(inline_comment_prefixes=';')
            config.optionxform = str
            config.read(os.path.join(os.path.dirname(__file__), f'models/{model_file}.ini'))
        except Exception:
            raise RuntimeError(f'Could not find specified model file: {model_file}')
        
        self.background_model = dict()
        for (background_component, scale_factor) in config.items('component_scalings'):
            self.background_model[background_component] = im.Spectrum(spectrum_file_folder=spectrum_file_folder,
                                                                      component=background_component,
                                                                      scale_factor=float(scale_factor))

        self.annual_cycles = dict()
        for (cycle, times) in config.items('annual_cycles'):
            times = times.split()
            assert len (times) == 2, 'Cycle can only be composed of two times (start and end)'
            t_start = pd.to_datetime(times[0])
            t_end = pd.to_datetime(times[1])
            self.annual_cycles[cycle] = (t_start, t_end)

        self.exposures = dict()
        for (cycle, exposure) in config.items('exposure_kgday'):
            self.exposures[cycle] = exposure

        assert self.annual_cycles.keys() == self.exposures.keys(), \
            'Must specify an exposure for each annual cycle'
        