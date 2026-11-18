"""Practical 8: hyperparameter tuning and model comparison with TensorBoard.

Fashion-MNIST MLP. One-factor-at-a-time search over learning rate, batch size and
number of layers around a baseline; every run is logged to TensorBoard (scalars + HParams)
and a short report is generated from the results.
View the logs with:  tensorboard --logdir logs/hparam_tuning
"""
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers

from .data import load_fashion, set_seed

LOG_ROOT = 'logs/hparam_tuning'
BASELINE = {'learning_rate': 1e-3, 'batch_size': 128, 'num_layers': 2}
VARIATIONS = {'learning_rate': [1e-2, 1e-4], 'batch_size': [32, 512], 'num_layers': [1, 3]}

HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_BATCH = hp.HParam('batch_size', hp.Discrete([32, 128, 512]))
HP_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 2, 3]))


def build_model(num_layers, learning_rate):
    model = keras.Sequential([keras.Input(shape=(784,))])
    for i in range(num_layers):
        model.add(layers.Dense(256 if i == 0 else 128, activation='relu'))
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def one_factor_runs(baseline=BASELINE, variations=VARIATIONS):
    """[('baseline', cfg), ('learning_rate=0.01', cfg), ...] - change one setting at a time."""
    runs = [('baseline', dict(baseline))]
    for key, values in variations.items():
        for v in values:
            cfg = dict(baseline)
            cfg[key] = v
            runs.append((f'{key}={v}', cfg))
    return runs


def init_logs(log_root=LOG_ROOT):
    shutil.rmtree(log_root, ignore_errors=True)
    with tf.summary.create_file_writer(log_root).as_default():
        hp.hparams_config(hparams=[HP_LR, HP_BATCH, HP_LAYERS],
                          metrics=[hp.Metric('best_val_accuracy', display_name='Best val accuracy'),
                                   hp.Metric('test_accuracy', display_name='Test accuracy')])


def run_experiment(name, cfg, train, val, test, epochs=15, log_root=LOG_ROOT):
    run_dir = os.path.join(log_root, name)
    set_seed()
    model = build_model(cfg['num_layers'], cfg['learning_rate'])
    t0 = time.time()
    h = model.fit(*train, validation_data=val, epochs=epochs, batch_size=cfg['batch_size'], verbose=0,
                  callbacks=[keras.callbacks.TensorBoard(log_dir=run_dir)]).history
    train_time = time.time() - t0
    test_acc = model.evaluate(*test, verbose=0)[1]
    best_val = max(h['val_accuracy'])
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams({HP_LR: cfg['learning_rate'], HP_BATCH: cfg['batch_size'], HP_LAYERS: cfg['num_layers']})
        tf.summary.scalar('best_val_accuracy', best_val, step=1)
        tf.summary.scalar('test_accuracy', test_acc, step=1)
    row = {'Run': name, **cfg, 'Params': model.count_params(), 'Best val acc': best_val,
           'Best epoch': int(np.argmax(h['val_accuracy'])) + 1, 'Final train acc': h['accuracy'][-1],
           'Final val loss': h['val_loss'][-1], 'Test acc': test_acc, 'Train time (s)': train_time}
    return h, row


def plot_factors(histories, variations=VARIATIONS):
    fig, axes = plt.subplots(2, len(variations), figsize=(6 * len(variations), 8), squeeze=False)
    for col, (key, values) in enumerate(variations.items()):
        names = [f'{key}={values[0]}', 'baseline'] + [f'{key}={v}' for v in values[1:]]
        for n in names:
            axes[0, col].plot(histories[n]['val_accuracy'], label=n)
            axes[1, col].plot(histories[n]['val_loss'], label=n)
        axes[0, col].set_title(f'{key}: validation accuracy'); axes[0, col].legend(fontsize=8)
        axes[1, col].set_title(f'{key}: validation loss'); axes[1, col].set_xlabel('Epoch')
        axes[1, col].legend(fontsize=8)
    plt.tight_layout(); plt.show()


def make_report(results, epochs, variations=VARIATIONS):
    """Plain-text comparison built from the results table, so the numbers always match the run."""
    r = results
    lines = [f'HYPERPARAMETER TUNING REPORT - Fashion-MNIST MLP ({epochs} epochs per run, {len(r)} runs)',
             '=' * 80,
             f"Baseline {BASELINE} -> test accuracy {r.loc['baseline', 'Test acc']:.4f}", '']
    for key, values in variations.items():
        names = [f'{key}={values[0]}', 'baseline'] + [f'{key}={v}' for v in values[1:]]
        cells = []
        for n in names:
            value = r.loc[n, key]
            cells.append(f'{key}={value}: test {r.loc[n, "Test acc"]:.4f}, {r.loc[n, "Train time (s)"]:.0f}s')
        spread = r.loc[names, 'Test acc'].max() - r.loc[names, 'Test acc'].min()
        lines += [f'{key}  (spread in test accuracy: {spread:.4f})', '   ' + ' | '.join(cells), '']
    best = r['Best val acc'].idxmax()
    lines += [f'Best configuration by validation accuracy: {best}',
              f"   best val acc {r.loc[best, 'Best val acc']:.4f} at epoch {int(r.loc[best, 'Best epoch'])}, "
              f"test acc {r.loc[best, 'Test acc']:.4f}",
              '',
              'The hyperparameter with the largest spread had the biggest effect on accuracy.',
              'Learning rate controls step size (too large: unstable; too small: slow).',
              'Batch size trades update noise against speed per epoch.',
              'Extra layers add capacity with diminishing returns for an MLP on raw pixels.']
    return '\n'.join(lines)


def run(epochs=15, log_root=LOG_ROOT, report_path='hyperparameter_report.txt'):
    train, val, test = load_fashion()
    init_logs(log_root)
    histories, rows = {}, []
    for name, cfg in one_factor_runs():
        h, row = run_experiment(name, cfg, train, val, test, epochs, log_root)
        histories[name] = h
        rows.append(row)
        print(f'{name:22s} best val {row["Best val acc"]:.4f} | test {row["Test acc"]:.4f} '
              f'| {row["Train time (s)"]:.0f}s')
    results = pd.DataFrame(rows).set_index('Run')
    print(results.round(4).to_string())
    plot_factors(histories)

    report = make_report(results, epochs)
    print(report)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\nReport saved to {report_path}. View runs with: tensorboard --logdir {log_root}')
    return results
