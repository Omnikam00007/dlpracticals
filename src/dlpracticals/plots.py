"""Plotting helpers shared by the practicals."""
import matplotlib.pyplot as plt
import numpy as np


def _h(history):
    return history.history if hasattr(history, 'history') else history


def plot_history(history, title=''):
    """Train vs validation loss and accuracy for one Keras history (or its .history dict)."""
    h = _h(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ['loss', 'accuracy']):
        ax.plot(h[key], label='Train')
        ax.plot(h[f'val_{key}'], label='Validation')
        ax.set_title(f'{title} {key.capitalize()}'.strip()); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout(); plt.show()


def compare_histories(histories, keys=('val_loss', 'val_accuracy'), title=''):
    """One line per model for each metric. `histories` maps name -> history."""
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 4.5))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for name, h in histories.items():
            ax.plot(_h(h)[key], label=name)
        ax.set_title(f'{title} {key}'.strip()); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def plot_confusion(y_true, y_pred, class_names, title='Confusion matrix'):
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title(title); plt.tight_layout(); plt.show()
