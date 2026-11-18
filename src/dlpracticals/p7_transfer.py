"""Practical 7: transfer learning with a pretrained ResNet50 on cats vs dogs (CO4).

Compares three freezing strategies. Each experiment first trains only the new head
(base frozen), then unfreezes the chosen stages and fine-tunes with a small learning rate.
Needs internet once to download the ImageNet weights; a GPU is strongly recommended.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from .data import dataset_path, set_seed

IMG_SIZE = (224, 224)
EXPERIMENTS = {
    'All frozen (feature extraction)': None,
    'Fine-tune conv5': 'conv5',
    'Fine-tune conv4 + conv5': 'conv4',
}


def load_cats_dogs(batch_size=32):
    """Returns (train_ds, val_ds, class_names) from cats_and_dogs_filtered/train and /validation."""
    root = dataset_path('cats_and_dogs_filtered.zip', 'cats_and_dogs_filtered', want_dir=True)
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(root, 'train'), image_size=IMG_SIZE, batch_size=batch_size, label_mode='binary', seed=42)
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(root, 'validation'), image_size=IMG_SIZE, batch_size=batch_size,
        label_mode='binary', shuffle=False)
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.cache().shuffle(1000).prefetch(autotune), val_ds.cache().prefetch(autotune), class_names


def build_model():
    """augmentation -> preprocess -> ResNet50 (frozen, inference mode) -> GAP -> Dropout -> Dense(1)."""
    base = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    augmentation = keras.Sequential([layers.RandomFlip('horizontal'),
                                     layers.RandomRotation(0.1),
                                     layers.RandomZoom(0.1)], name='augmentation')
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = preprocess_input(x)
    x = base(x, training=False)          # keeps BatchNorm statistics fixed during fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs), base


def set_trainable(base, unfreeze_from):
    """None freezes the whole base; 'conv5' / 'conv4' unfreezes from that stage on (BatchNorm stays frozen)."""
    if unfreeze_from is None:
        base.trainable = False
        return
    base.trainable = True
    start = next(i for i, l in enumerate(base.layers) if l.name.startswith(unfreeze_from))
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)


def count_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def run_experiment(train_ds, val_ds, unfreeze_from, head_epochs=5, ft_epochs=5):
    set_seed()
    model, base = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, verbose=2)

    set_trainable(base, unfreeze_from)
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    print('Trainable parameters in stage 2:', f'{count_trainable(model):,}')
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs + ft_epochs,
                   initial_epoch=head_epochs, verbose=2)
    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    return model, history, h1.history['val_accuracy'][-1]


def show_predictions(model, val_ds, class_names, title=''):
    images, labels = next(iter(val_ds.unbatch().shuffle(1000, seed=1).batch(12)))
    probs = model.predict(images, verbose=0).ravel()
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().astype('uint8')); ax.axis('off')
        pred, true = int(probs[i] >= 0.5), int(labels[i][0])
        ax.set_title(f'{class_names[pred]} ({max(probs[i], 1 - probs[i]):.2f})\ntrue: {class_names[true]}',
                     color='green' if pred == true else 'red', fontsize=9)
    plt.suptitle(f'{title}: predictions'); plt.tight_layout(); plt.show()


def run(head_epochs=5, ft_epochs=5, experiments=EXPERIMENTS):
    train_ds, val_ds, class_names = load_cats_dogs()
    print('Classes:', class_names)
    models, histories, rows = {}, {}, []
    for name, unfreeze_from in experiments.items():
        print(f'\n===== {name} =====')
        model, history, head_acc = run_experiment(train_ds, val_ds, unfreeze_from, head_epochs, ft_epochs)
        models[name], histories[name] = model, history
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        rows.append({'Experiment': name, 'Trainable params (stage 2)': count_trainable(model),
                     'Val acc after head only': head_acc, 'Final val acc': val_acc, 'Final val loss': val_loss})

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for name, h in histories.items():
        axes[0].plot(h['val_accuracy'], marker='o', label=name)
        axes[1].plot(h['val_loss'], marker='o', label=name)
    for ax, t in zip(axes, ['Validation accuracy', 'Validation loss']):
        ax.axvline(head_epochs - 0.5, color='gray', ls='--')   # fine-tuning starts here
        ax.set_title(t); ax.set_xlabel('Epoch'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

    summary = pd.DataFrame(rows).set_index('Experiment')
    print(summary.round(4).to_string())
    best = summary['Final val acc'].idxmax()
    show_predictions(models[best], val_ds, class_names, best)
    return summary
