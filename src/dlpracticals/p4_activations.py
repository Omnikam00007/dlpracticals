"""Practical 4: Sigmoid vs Tanh vs ReLU on MNIST, training speed and gradient flow (CO2).

A deep MLP (6 hidden layers) is trained with SGD + momentum so vanishing gradients are
visible; a 20-layer demo shows vanishing vs exploding gradients at initialisation.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import load_mnist, set_seed

ACTIVATIONS = {'Sigmoid': 'sigmoid', 'Tanh': 'tanh', 'ReLU': 'relu'}
HIDDEN = (128,) * 6
_loss_fn = keras.losses.SparseCategoricalCrossentropy()


def plot_activation_functions():
    z = np.linspace(-6, 6, 400)
    sig = 1 / (1 + np.exp(-z))
    funcs = {'Sigmoid': (sig, sig * (1 - sig)),
             'Tanh': (np.tanh(z), 1 - np.tanh(z) ** 2),
             'ReLU': (np.maximum(0, z), (z > 0).astype(float))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, (f, df) in funcs.items():
        axes[0].plot(z, f, label=name); axes[1].plot(z, df, label=name)
    axes[0].set_title('Activation f(z)'); axes[1].set_title("Derivative f'(z)")
    for ax in axes:
        ax.axhline(0, color='gray', lw=0.5); ax.legend(); ax.set_xlabel('z')
    plt.tight_layout(); plt.show()


def build_mlp(activation, initializer='glorot_uniform', hidden=HIDDEN):
    model = keras.Sequential([keras.Input(shape=(784,))])
    for units in hidden:
        model.add(layers.Dense(units, activation=activation, kernel_initializer=initializer))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def layer_grad_norms(model, x, y):
    """||dLoss/dW|| for every Dense kernel, input side first."""
    dense = [l for l in model.layers if isinstance(l, layers.Dense)]
    with tf.GradientTape() as tape:
        loss = _loss_fn(y, model(x, training=False))
    grads = tape.gradient(loss, [l.kernel for l in dense])
    return [float(tf.norm(g)) for g in grads]


class GradFlowAndTime(keras.callbacks.Callback):
    """Records per-layer gradient norms (at init and after each epoch) and seconds per epoch."""

    def __init__(self, x_probe, y_probe):
        super().__init__()
        self.x_probe, self.y_probe = tf.constant(x_probe), tf.constant(y_probe)

    def on_train_begin(self, logs=None):
        self.grad_norms = [layer_grad_norms(self.model, self.x_probe, self.y_probe)]
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self._t0)
        self.grad_norms.append(layer_grad_norms(self.model, self.x_probe, self.y_probe))


def compare_activations(train, val, test, activations=ACTIVATIONS, epochs=15, lr=0.01):
    """Returns (histories, recorders, test_acc) keyed by activation name."""
    (x_tr, y_tr) = train
    histories, recorders, test_acc = {}, {}, {}
    for name, act in activations.items():
        set_seed()
        model = build_mlp(act)
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                      loss=_loss_fn, metrics=['accuracy'])
        rec = GradFlowAndTime(x_tr[:1024], y_tr[:1024])
        histories[name] = model.fit(x_tr, y_tr, validation_data=val, epochs=epochs,
                                    batch_size=128, callbacks=[rec], verbose=0).history
        recorders[name] = rec
        test_acc[name] = model.evaluate(*test, verbose=0)[1]
        print(f'{name:8s} final val acc {histories[name]["val_accuracy"][-1]:.4f} | test acc {test_acc[name]:.4f}')
    return histories, recorders, test_acc


def speed_table(histories, recorders, test_acc, target=0.95):
    def epochs_to(h):
        hits = [i + 1 for i, a in enumerate(h['val_accuracy']) if a >= target]
        return hits[0] if hits else np.nan
    return pd.DataFrame({
        'Avg sec / epoch': {n: np.mean(r.epoch_times) for n, r in recorders.items()},
        f'Epochs to {target:.0%} val acc': {n: epochs_to(h) for n, h in histories.items()},
        'Final val acc': {n: h['val_accuracy'][-1] for n, h in histories.items()},
        'Test acc': test_acc,
    })


def plot_gradient_flow(recorders):
    n_layers = len(next(iter(recorders.values())).grad_norms[0])
    ids = np.arange(1, n_layers + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for name, rec in recorders.items():
        axes[0].semilogy(ids, rec.grad_norms[0], marker='o', label=name)
        axes[1].semilogy(ids, rec.grad_norms[-1], marker='o', label=name)
    axes[0].set_title('Gradient norm per layer - at initialisation')
    axes[1].set_title('Gradient norm per layer - after training')
    for ax in axes:
        ax.set_xlabel('Layer (1 = input side)'); ax.set_xticks(ids); ax.legend()
    axes[0].set_ylabel('||dL/dW|| (log scale)')
    plt.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, len(recorders), figsize=(6 * len(recorders), 4.5), sharey=True, squeeze=False)
    for ax, (name, rec) in zip(axes[0], recorders.items()):
        g = np.array(rec.grad_norms)                  # (epochs + 1, layers)
        for layer in range(n_layers):
            ax.semilogy(g[:, layer], label=f'Layer {layer + 1}')
        ax.set_title(f'{name}: gradient norm during training'); ax.set_xlabel('Epoch')
    axes[0, -1].legend(fontsize=8)
    plt.tight_layout(); plt.show()


def deep_gradient_demo(x, y, depth=20):
    """Gradient norms at initialisation in a very deep network: vanishing, exploding and healthy cases."""
    hidden = (64,) * depth
    cases = {
        'Sigmoid (Glorot init) - vanishing': build_mlp('sigmoid', 'glorot_uniform', hidden),
        'ReLU (N(0,1) init) - exploding': build_mlp('relu', keras.initializers.RandomNormal(stddev=1.0), hidden),
        'ReLU (He init) - healthy': build_mlp('relu', 'he_normal', hidden),
    }
    x, y = tf.constant(x), tf.constant(y)
    plt.figure(figsize=(9, 4.5))
    for name, m in cases.items():
        norms = layer_grad_norms(m, x, y)
        plt.semilogy(range(1, len(norms) + 1), norms, marker='.', label=name)
        print(f'{name:36s} layer1 = {norms[0]:.2e} | last layer = {norms[-1]:.2e}')
    plt.xlabel('Layer (1 = input side)'); plt.ylabel('||dL/dW|| (log scale)')
    plt.title(f'Gradient norm per layer in a {depth}-layer network (at initialisation)'); plt.legend(); plt.show()


def run(epochs=15):
    plot_activation_functions()
    train, val, test = load_mnist()
    histories, recorders, test_acc = compare_activations(train, val, test, epochs=epochs)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for name, h in histories.items():
        axes[0].plot(h['loss'], label=name)
        axes[1].plot(h['accuracy'], label=name)
        axes[2].plot(h['val_accuracy'], label=name)
    for ax, t in zip(axes, ['Training loss', 'Training accuracy', 'Validation accuracy']):
        ax.set_title(t); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout(); plt.show()

    speed = speed_table(histories, recorders, test_acc)
    print(speed.round(4).to_string())
    plot_gradient_flow(recorders)
    deep_gradient_demo(train[0][:1024], train[1][:1024])
    return speed
