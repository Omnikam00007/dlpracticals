"""Deep learning practicals 1-8.

Quick start (Jupyter / Colab):

    import dlpracticals as dp
    dp.prepare_data('Dataset')     # extract all dataset zips into ./data (once)
    dp.notebook(3)                 # write Practical_3.ipynb, one cell per step
    dp.fill(3)                     # or: put practical 3 into the next cell of this notebook
    dp.show_code(3)                # or: just display the code

Full versions with extra analysis live in the modules p1_perceptron ... p8_tuning,
each with a run() that performs the whole practical.
"""
import importlib

from .data import get_data_dir, prepare_data, set_data_dir
from .notebooks import all_notebooks, code, fill, notebook, show_code

__version__ = '1.1.0'

PRACTICALS = {
    1: 'p1_perceptron',
    2: 'p2_mlp',
    3: 'p3_regularization',
    4: 'p4_activations',
    5: 'p5_cnn',
    6: 'p6_lstm',
    7: 'p7_transfer',
    8: 'p8_tuning',
}

__all__ = ['prepare_data', 'notebook', 'all_notebooks', 'fill', 'show_code', 'code',
           'set_data_dir', 'get_data_dir', 'run_practical', 'PRACTICALS']


def run_practical(number, **kwargs):
    """Run the full version of practical `number` (1-8)."""
    if number not in PRACTICALS:
        raise ValueError(f'Practical must be 1-8, got {number}')
    return importlib.import_module(f'.{PRACTICALS[number]}', __name__).run(**kwargs)


def __getattr__(name):
    if name in PRACTICALS.values():
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
