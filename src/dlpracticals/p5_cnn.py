"""Practical 5: simple CNN for CIFAR-10, compared with an MLP, with data augmentation (CO3).

Images are read from the PNG folders in `cifar (1).zip` (cifar10/train/<class>, cifar10/test/<class>).
A GPU is strongly recommended.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import dataset_path, set_seed
from .plots import plot_confusion


def load_cifar10(batch_size=64, val_split=0.1, seed=42):
    """Returns (train_ds, val_ds, test_ds, class_names) as batched tf.data datasets of uint8-range images."""
    root = dataset_path('cifar (1).zip', 'cifar10', want_dir=True)
    common = dict(image_size=(32, 32), batch_size=batch_size, label_mode='int')
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(root, 'train'), validation_split=val_split, subset='training', seed=seed, **common)
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(root, 'train'), validation_split=val_split, subset='validation', seed=seed, **common)
    test_ds = keras.utils.image_dataset_from_directory(os.path.join(root, 'test'), shuffle=False, **common)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (train_ds.cache().shuffle(5000).prefetch(autotune), val_ds.cache().prefetch(autotune),
            test_ds.cache().prefetch(autotune), class_names)


def make_augmentation():
    return keras.Sequential([layers.RandomFlip('horizontal'),
                             layers.RandomTranslation(0.1, 0.1),
                             layers.RandomRotation(0.05)], name='augmentation')


def build_mlp():
    return keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Rescaling(1. / 255),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name='MLP')


def build_cnn(augment=False):
    """2 conv layers (+ max pooling) followed by fully connected layers."""
    model = keras.Sequential([keras.Input(shape=(32, 32, 3))], name='CNN_augmented' if augment else 'CNN')
    if augment:
        model.add(make_augmentation())
    model.add(layers.Rescaling(1. / 255))
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def show_augmentation(train_ds, class_names, n=8):
    images, labels = next(iter(train_ds))
    aug = make_augmentation()
    fig, axes = plt.subplots(1, n, figsize=(14, 2.2))
    for ax in axes:
        ax.imshow(aug(images[:1], training=True)[0].numpy().astype('uint8')); ax.axis('off')
    plt.suptitle(f'Augmented versions of one "{class_names[int(labels[0])]}" image'); plt.show()


def compare(train_ds, val_ds, test_ds, epochs=20):
    """Train MLP, CNN and CNN + augmentation; returns (models, histories, comparison DataFrame)."""
    set_seed()
    models = {'MLP': build_mlp(), 'CNN': build_cnn(), 'CNN + augmentation': build_cnn(augment=True)}
    histories, rows = {}, []
    for name, model in models.items():
        print(f'\n===== {name} =====')
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2).history
        histories[name] = h
        loss, acc = model.evaluate(test_ds, verbose=0)
        rows.append({'Model': name, 'Parameters': model.count_params(), 'Train acc': h['accuracy'][-1],
                     'Val acc': h['val_accuracy'][-1], 'Test acc': acc, 'Test loss': loss})
    return models, histories, pd.DataFrame(rows).set_index('Model')


def plot_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, h in histories.items():
        for ax, key in zip(axes, ['accuracy', 'loss']):
            line, = ax.plot(h[f'val_{key}'], label=f'{name} (val)')
            ax.plot(h[key], '--', color=line.get_color(), alpha=0.6, label=f'{name} (train)')
    for ax, t in zip(axes, ['Accuracy', 'Loss']):
        ax.set_title(t); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def show_predictions(model, test_ds, class_names, title=''):
    images, labels = next(iter(test_ds))
    labels = labels.numpy()
    pred = model.predict(images, verbose=0).argmax(axis=1)
    fig, axes = plt.subplots(2, 8, figsize=(15, 4.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype('uint8')); ax.axis('off')
        ax.set_title(f'P: {class_names[pred[i]]}\nT: {class_names[labels[i]]}', fontsize=8,
                     color='green' if pred[i] == labels[i] else 'red')
    plt.suptitle(f'{title}: predictions (green = correct)'); plt.tight_layout(); plt.show()


def run(epochs=20):
    train_ds, val_ds, test_ds, class_names = load_cifar10()
    print('Classes:', class_names)
    show_augmentation(train_ds, class_names)
    models, histories, comparison = compare(train_ds, val_ds, test_ds, epochs=epochs)
    plot_curves(histories)
    print(comparison.round(4).to_string())

    best_name = comparison['Test acc'].idxmax()
    best = models[best_name]
    show_predictions(best, test_ds, class_names, best_name)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    plot_confusion(y_true, best.predict(test_ds, verbose=0).argmax(axis=1), class_names,
                   f'{best_name}: confusion matrix')
    return comparison
