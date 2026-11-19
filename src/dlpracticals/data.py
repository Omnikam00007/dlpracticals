"""Dataset location and loaders shared by the practicals.

The datasets are the zips inside the "Dataset" folder:
Handwritten.zip, Fashion.zip, IMDB.zip, cifar (1).zip, cats_and_dogs_filtered.zip.
Each zip is extracted once into the extract folder on first use.
"""
import glob
import io
import os
import shutil
import zipfile

_DATA_DIR = os.environ.get('DL_DATA_DIR', 'Dataset')
_EXTRACT_DIR = os.environ.get('DL_EXTRACT_DIR', 'data')


def set_data_dir(data_dir, extract_dir=None):
    """Point the library at the folder that holds the dataset zips."""
    global _DATA_DIR, _EXTRACT_DIR
    _DATA_DIR = data_dir
    if extract_dir is not None:
        _EXTRACT_DIR = extract_dir


def get_data_dir():
    return _DATA_DIR


def _inner_zips(source):
    """Yield (name, ZipFile) for every dataset zip in a folder, or inside the downloaded outer zip."""
    if os.path.isfile(source) and source.lower().endswith('.zip'):
        outer = zipfile.ZipFile(source)
        for name in outer.namelist():
            if name.lower().endswith('.zip'):
                yield os.path.basename(name), zipfile.ZipFile(io.BytesIO(outer.read(name)))
        return
    if not os.path.isdir(source):
        raise FileNotFoundError(f'{source} not found. Pass the Dataset folder or the downloaded Dataset zip.')
    for path in sorted(glob.glob(os.path.join(source, '**', '*.zip'), recursive=True)):
        yield os.path.basename(path), zipfile.ZipFile(path)


def prepare_data(source=None, out='data'):
    """Extract all dataset zips into one folder and return its path.

    `source` is the Dataset folder (holding Handwritten.zip, Fashion.zip, ...) or the downloaded
    outer zip. CSV files land directly in `out` (out/mnist_train.csv, out/IMDB Dataset.csv, ...);
    image folders keep their structure (out/cifar10/train, out/cats_and_dogs_filtered/train).
    Zips already extracted are skipped, so calling this again is instant.
    """
    source = source or _DATA_DIR
    os.makedirs(out, exist_ok=True)
    for zip_name, archive in _inner_zips(source):
        marker = os.path.join(out, f'.extracted_{zip_name}')
        if os.path.exists(marker):
            continue
        print('Extracting', zip_name, '...')
        with archive:
            for member in archive.infolist():
                name = member.filename
                if member.is_dir() or 'ubyte' in name:
                    continue
                target = os.path.join(out, os.path.basename(name) if name.lower().endswith('.csv') else name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(member) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
        open(marker, 'w').close()
    return out


def dataset_path(zip_name, name, want_dir=False):
    """Unzip DATA_DIR/zip_name once and return the path of `name` inside it."""
    out = os.path.join(_EXTRACT_DIR, os.path.splitext(zip_name)[0])
    if not os.path.isdir(out):
        src = os.path.join(_DATA_DIR, zip_name)
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f'{src} not found. Call dlpracticals.set_data_dir(...) with the folder holding {zip_name}.')
        print('Extracting', zip_name, '...')
        with zipfile.ZipFile(src) as z:
            z.extractall(out)
    check = os.path.isdir if want_dir else os.path.isfile
    hits = [p for p in glob.glob(os.path.join(out, '**', name), recursive=True) if check(p)]
    if not hits:
        raise FileNotFoundError(f'{name} not found inside {zip_name}')
    return sorted(hits, key=len)[0]


def load_image_csv(path):
    """Read a label + 784-pixel CSV. Returns x in [0, 1] (float32) and integer labels y."""
    import pandas as pd
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy('int64')                 # labels stay whole numbers
    x = df.iloc[:, 1:].to_numpy('float32') / 255.0      # only the pixels are scaled
    return x, y


def _split(x_all, y_all, x_test, y_test, n_train):
    return (x_all[:n_train], y_all[:n_train]), (x_all[50000:], y_all[50000:]), (x_test, y_test)


def load_fashion(n_train=50000):
    """Fashion-MNIST as (train, val, test) tuples of (x, y). Validation = last 10,000 training rows."""
    x_all, y_all = load_image_csv(dataset_path('Fashion.zip', 'fashion-mnist_train.csv'))
    x_test, y_test = load_image_csv(dataset_path('Fashion.zip', 'fashion-mnist_test.csv'))
    return _split(x_all, y_all, x_test, y_test, min(n_train, 50000))


def load_mnist(n_train=50000):
    """MNIST digits as (train, val, test) tuples of (x, y)."""
    x_all, y_all = load_image_csv(dataset_path('Handwritten.zip', 'mnist_train.csv'))
    x_test, y_test = load_image_csv(dataset_path('Handwritten.zip', 'mnist_test.csv'))
    return _split(x_all, y_all, x_test, y_test, min(n_train, 50000))


FASHION_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def set_seed(seed=42):
    import numpy as np
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
