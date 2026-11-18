"""Practical 3: L2 regularization and Dropout on Fashion-MNIST (CO2).

A large MLP trained on only 10,000 images for 40 epochs overfits clearly;
the same model with L2, Dropout or both is compared against it.
"""
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .data import load_fashion, set_seed
from .plots import compare_histories

CONFIGS = {
    'No regularization': dict(l2=0.0, dropout=0.0),
    'L2 (1e-3)': dict(l2=1e-3, dropout=0.0),
    'Dropout (0.5)': dict(l2=0.0, dropout=0.5),
    'L2 (1e-4) + Dropout (0.3)': dict(l2=1e-4, dropout=0.3),
}


def build_model(l2=0.0, dropout=0.0, hidden=(512, 512, 256)):
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = keras.Sequential([keras.Input(shape=(784,))])
    for units in hidden:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def compare(train, val, test, configs=CONFIGS, epochs=40, batch_size=128):
    """Train one model per config; returns (summary DataFrame, histories)."""
    histories, rows = {}, []
    for name, cfg in configs.items():
        set_seed()
        model = build_model(**cfg)
        h = model.fit(*train, validation_data=val, epochs=epochs, batch_size=batch_size, verbose=0).history
        histories[name] = h
        rows.append({'Model': name,
                     'Final train acc': h['accuracy'][-1],
                     'Final val acc': h['val_accuracy'][-1],
                     'Gap (train - val)': h['accuracy'][-1] - h['val_accuracy'][-1],
                     'Min val loss': min(h['val_loss']),
                     'Final val loss': h['val_loss'][-1],
                     'Test acc': model.evaluate(*test, verbose=0)[1]})
        print(f'{name:28s} train {h["accuracy"][-1]:.4f} | val {h["val_accuracy"][-1]:.4f}')
    return pd.DataFrame(rows).set_index('Model'), histories


def plot_train_vs_val(histories):
    """Grid: one column per model, loss on top, accuracy below."""
    fig, axes = plt.subplots(2, len(histories), figsize=(5 * len(histories), 8), sharex=True, squeeze=False)
    for col, (name, h) in enumerate(histories.items()):
        for row, key in enumerate(['loss', 'accuracy']):
            ax = axes[row, col]
            ax.plot(h[key], label='Train'); ax.plot(h[f'val_{key}'], label='Validation')
            ax.set_title(f'{name}\n{key}' if row == 0 else key); ax.legend()
        axes[1, col].set_xlabel('Epoch')
    plt.tight_layout(); plt.show()


def run(epochs=40, n_train=10000):
    train, val, test = load_fashion(n_train=n_train)     # small training set to provoke overfitting
    summary, histories = compare(train, val, test, epochs=epochs)
    plot_train_vs_val(histories)
    compare_histories(histories, title='Validation')
    print(summary.round(4).to_string())
    summary[['Final train acc', 'Final val acc', 'Test acc']].plot.bar(figsize=(10, 4.5), rot=0)
    plt.ylim(0.75, 1.0); plt.title('Accuracy comparison'); plt.tight_layout(); plt.show()
    return summary
