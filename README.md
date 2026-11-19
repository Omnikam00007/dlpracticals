# dlpracticals

Deep learning practicals 1-8 (TensorFlow / Keras) that write themselves into Jupyter notebooks.

| # | Practical | Data file(s) used |
|---|---|---|
| 1 | Perceptron from scratch (NumPy) | none |
| 2 | Feedforward network (MLP), tune layers and activations | fashion-mnist_train/test.csv |
| 3 | L2 and Dropout regularization | fashion-mnist_train/test.csv |
| 4 | Sigmoid vs Tanh vs ReLU, gradient flow | mnist_train/test.csv |
| 5 | CNN vs MLP with data augmentation | cifar10/ |
| 6 | LSTM sentiment classification | IMDB Dataset.csv |
| 7 | Transfer learning with ResNet50 | cats_and_dogs_filtered/ |
| 8 | Hyperparameter tuning + TensorBoard | fashion-mnist_train/test.csv |

## Install

```
pip install https://github.com/Omnikam00007/dlpracticals/archive/refs/heads/main.zip
```

This installs only the practical code (a few KB). No datasets are included and no other libraries are
installed; the PC needs TensorFlow, pandas, matplotlib and scikit-learn already to *run* the practicals.
To install those too: `pip install "dlpracticals[full] @ https://github.com/Omnikam00007/dlpracticals/archive/refs/heads/main.zip"`.

## Use in Jupyter or Colab

```python
import dlpracticals as dp

dp.prepare_data('Dataset')   # extracts every dataset zip into ./data (only the first time)
dp.notebook(3)               # creates Practical_3.ipynb with the code split into cells
dp.all_notebooks('notebooks')  # creates all 8 notebooks in a folder
dp.fill(3)                   # puts practical 3's code into the next cell of the open notebook
dp.show_code(3)              # just displays the code
dp.show_code(3, save_to='practical3.py')
```

`'Dataset'` is the folder holding `Handwritten.zip`, `Fashion.zip`, `IMDB.zip`, `cifar (1).zip` and
`cats_and_dogs_filtered.zip`. You can also pass the downloaded zip itself:
`dp.prepare_data('Dataset-20261112T113230Z-1-001.zip')`.

Every generated notebook starts with:

```python
import dlpracticals as dp
DATA = dp.prepare_data('Dataset')
```

and then reads its files from `DATA`, e.g. `pd.read_csv(f'{DATA}/fashion-mnist_train.csv')`.

## Command line

```
python -m dlpracticals prepare --source Dataset
python -m dlpracticals notebook all --out notebooks
python -m dlpracticals show 2
```

## Notes

- Practicals 5 and 7 need a GPU (Colab: Runtime -> Change runtime type -> GPU).
- Practical 7 downloads the ResNet50 weights (~100 MB) the first time.
- Longer versions with extra analysis are in `dlpracticals.p1_perceptron` ... `p8_tuning` (`run()` in each),
  or `python -m dlpracticals run 3`.
