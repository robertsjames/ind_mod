"""
01/24, R. S. James
"""

import ind_mod as im
import sys, os
import configparser

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

        for (background_component, _) in config.items('components'):
            self.background_model[background_component] = im.Spectrum(spectrum_file_folder=spectrum_file_folder,
                                                                      component=background_component)
        