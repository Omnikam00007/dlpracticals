# dlpracticals

Deep learning practicals 1-8 as one Python library (TensorFlow / Keras).

| Module | Practical | Dataset |
|---|---|---|
| `p1_perceptron` | 1. Perceptron from scratch (NumPy) | AND / OR / XOR + synthetic |
| `p2_mlp` | 2. Feedforward network, tune layers and activations | Fashion.zip |
| `p3_regularization` | 3. L2 and Dropout | Fashion.zip |
| `p4_activations` | 4. Sigmoid / Tanh / ReLU, gradient flow | Handwritten.zip (MNIST) |
| `p5_cnn` | 5. CNN vs MLP, data augmentation | cifar (1).zip |
| `p6_lstm` | 6. LSTM sentiment classifier | IMDB.zip |
| `p7_transfer` | 7. Transfer learning with ResNet50 | cats_and_dogs_filtered.zip |
| `p8_tuning` | 8. Hyperparameter tuning + TensorBoard | Fashion.zip |

## Install

From this folder (the one containing `pyproject.toml`):

```
pip install -e .
```

On Google Colab, upload the `dlpracticals` folder and run `!pip install -e /content/dlpracticals`.

## Datasets

Extract `Dataset-20261112T113230Z-1-001.zip` once to get the `Dataset` folder.
Leave the inner zips as they are; each one is extracted into `./data` the first time it's needed.

Tell the library where that folder is, in any one of these ways:
- `dlpracticals.set_data_dir('path/to/Dataset')` in Python
- `--data-dir path/to/Dataset` on the command line
- the `DL_DATA_DIR` environment variable
- or put the `Dataset` folder in the directory you run from (the default)

## Use

Run a whole practical:

```python
import dlpracticals as dp
dp.set_data_dir('Dataset')
dp.run_practical(2)          # or dp.p2_mlp.run()
```

```
python -m dlpracticals 5 --data-dir Dataset
```

Display the code of a practical on screen:

```python
dp.show_code(1)                        # practical 1's code (syntax-highlighted in Jupyter/Colab)
dp.show_code(5, helpers=True)          # also show the shared data.py / plots.py code
dp.show_code(3, save_to='p3.py')       # also save it to a file you can open and edit
```

```
python -m dlpracticals show 1
python -m dlpracticals show 5 --helpers --save p5.py
```

Or use the building blocks yourself:

```python
from dlpracticals import p1_perceptron as p1
p = p1.Perceptron(learning_rate=0.1, epochs=20, verbose=True).fit(p1.X_GATE, p1.GATES['AND'])
p1.plot_decision_boundary(p, p1.X_GATE, p1.GATES['AND'], 'AND')

from dlpracticals import p2_mlp, data
train, val, test = data.load_fashion()
model = p2_mlp.build_mlp((256, 128), 'relu')
model.fit(*train, validation_data=val, epochs=10)
```

`run()` in every module accepts the epoch counts as arguments, e.g. `dp.p5_cnn.run(epochs=5)` for a quick check.
Practicals 5 and 7 need a GPU to finish in reasonable time; practical 7 downloads the ResNet50 weights (~100 MB) once.
Practical 1 works without TensorFlow.
