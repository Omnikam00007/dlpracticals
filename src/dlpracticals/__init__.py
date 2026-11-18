"""Deep learning practicals 1-8 as a reusable library.

Each practical is its own module with reusable building blocks and a ``run()``
function that performs the whole practical:

    p1_perceptron      Perceptron from scratch (NumPy)
    p2_mlp             Feedforward network on Fashion-MNIST
    p3_regularization  L2 and Dropout comparison
    p4_activations     Sigmoid / Tanh / ReLU and gradient flow
    p5_cnn             CNN vs MLP on CIFAR-10 with augmentation
    p6_lstm            LSTM sentiment classifier on IMDB
    p7_transfer        Transfer learning with ResNet50 (cats vs dogs)
    p8_tuning          Hyperparameter tuning with TensorBoard

Modules are imported on demand so that ``import dlpracticals`` stays fast and
practical 1 works without TensorFlow installed.

Example:
    import dlpracticals as dp
    dp.set_data_dir('path/to/Dataset')
    dp.p2_mlp.run()
"""
import importlib
import os

from .data import set_data_dir, get_data_dir

__version__ = '1.0.0'

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

__all__ = ['set_data_dir', 'get_data_dir', 'run_practical', 'show_code', 'PRACTICALS', *PRACTICALS.values()]


def _source(name):
    # read the file directly so showing code never imports TensorFlow
    path = os.path.join(os.path.dirname(__file__), f'{name}.py')
    with open(path, encoding='utf-8') as f:
        return f.read()


def show_code(number, helpers=False, save_to=None):
    """Display the full source code of practical `number` (1-8).

    helpers=True also shows the shared data.py / plots.py code it uses.
    save_to='file.py' additionally writes the code to that file.
    In Jupyter/Colab the code is shown with syntax highlighting; elsewhere it is printed.
    """
    if number not in PRACTICALS:
        raise ValueError(f'Practical must be 1-8, got {number}')
    names = [PRACTICALS[number]] + (['data', 'plots'] if helpers else [])
    code = '\n\n'.join(f'# {"=" * 70}\n# {n}.py\n# {"=" * 70}\n{_source(n)}' for n in names)
    if save_to:
        with open(save_to, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f'Saved to {save_to}')
    try:
        from IPython import get_ipython
        from IPython.display import Code, display
        if get_ipython() is not None:
            display(Code(code, language='python'))
            return
    except ImportError:
        pass
    print(code)


def run_practical(number, **kwargs):
    """Run practical `number` (1-8). Extra keyword arguments go to that module's run()."""
    if number not in PRACTICALS:
        raise ValueError(f'Practical must be 1-8, got {number}')
    return importlib.import_module(f'.{PRACTICALS[number]}', __name__).run(**kwargs)


def __getattr__(name):
    # lazy import: dlpracticals.p5_cnn loads TensorFlow only when first used
    if name in PRACTICALS.values():
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
