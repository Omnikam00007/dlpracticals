"""Practical 2: feedforward neural network (MLP) on Fashion-MNIST (CO1).

Builds an MLP in Keras, reports loss/accuracy curves and test accuracy, and tunes
the hidden layers and activation function.
"""
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

from .data import FASHION_CLASSES, load_fashion, set_seed
from .plots import compare_histories, plot_confusion, plot_history

ARCHITECTURES = {'1 layer (128)': (128,),
                 '2 layers (256-128)': (256, 128),
                 '3 layers (512-256-128)': (512, 256, 128)}
ACTIVATIONS = ['relu', 'tanh', 'sigmoid']


def build_mlp(hidden_units=(256, 128), activation='relu', learning_rate=1e-3, n_inputs=784, n_classes=10):
    model = keras.Sequential([keras.Input(shape=(n_inputs,))])
    for units in hidden_units:
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(n_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def tune(train, val, architectures=ARCHITECTURES, activations=ACTIVATIONS, epochs=10, batch_size=128):
    """Train every architecture x activation; returns (results DataFrame sorted best-first, histories)."""
    (x_tr, y_tr), (x_val, y_val) = train, val
    rows, histories = [], {}
    for arch_name, units in architectures.items():
        for act in activations:
            set_seed()
            model = build_mlp(units, act)
            h = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                          epochs=epochs, batch_size=batch_size, verbose=0).history
            histories[(arch_name, act)] = h
            rows.append({'Architecture': arch_name, 'Activation': act, 'Params': model.count_params(),
                         'Train acc': h['accuracy'][-1], 'Best val acc': max(h['val_accuracy']),
                         'Final val loss': h['val_loss'][-1]})
            print(f'{arch_name:24s} {act:8s} best val acc = {max(h["val_accuracy"]):.4f}')
    return pd.DataFrame(rows).sort_values('Best val acc', ascending=False), histories


def run(epochs=20, tune_epochs=10):
    set_seed()
    train, val, (x_test, y_test) = load_fashion()

    # baseline
    model = build_mlp()
    model.summary()
    history = model.fit(*train, validation_data=val, epochs=epochs, batch_size=128, verbose=2)
    plot_history(history, 'Baseline MLP (256-128, ReLU) -')
    print('Baseline test accuracy: {:.4f}'.format(model.evaluate(x_test, y_test, verbose=0)[1]))

    # tuning
    results, histories = tune(train, val, epochs=tune_epochs)
    print(results.round(4).to_string(index=False))
    for arch_name in ARCHITECTURES:
        compare_histories({act: histories[(arch_name, act)] for act in ACTIVATIONS},
                          keys=('val_accuracy',), title=arch_name)

    # retrain the best configuration
    best = results.iloc[0]
    print('Best configuration:', best['Architecture'], '+', best['Activation'])
    set_seed()
    best_model = build_mlp(ARCHITECTURES[best['Architecture']], best['Activation'])
    best_hist = best_model.fit(*train, validation_data=val, epochs=epochs, batch_size=128, verbose=0)
    plot_history(best_hist, 'Best MLP -')
    print('Best model test accuracy: {:.4f}'.format(best_model.evaluate(x_test, y_test, verbose=0)[1]))
    plot_confusion(y_test, best_model.predict(x_test, verbose=0).argmax(axis=1), FASHION_CLASSES)
    return best_model, results
