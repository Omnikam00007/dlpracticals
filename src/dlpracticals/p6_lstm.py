"""Practical 6: LSTM for binary sentiment classification on IMDB (CO3).

Reads `IMDB Dataset.csv` (columns: review, sentiment), vectorises the text,
trains Embedding -> LSTM -> Dense(sigmoid), and shows example predictions.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .data import dataset_path, set_seed
from .plots import plot_confusion, plot_history

VOCAB, MAX_LEN = 20000, 250


def clean(text):
    text = re.sub(r'<br\s*/?>', ' ', text)          # HTML line breaks in the reviews
    text = re.sub(r"[^a-zA-Z']", ' ', text)          # keep letters and apostrophes
    return re.sub(r'\s+', ' ', text).strip().lower()


def load_imdb(seed=42):
    """Returns (train, val, test) tuples of (texts, labels): 70% / 10% / 20%, stratified."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(dataset_path('IMDB.zip', 'IMDB Dataset.csv'))
    texts = df['review'].apply(clean).values
    labels = (df['sentiment'] == 'positive').astype('int32').values
    x_temp, x_test, y_temp, y_test = train_test_split(texts, labels, test_size=0.2,
                                                      stratify=labels, random_state=seed)
    x_tr, x_val, y_tr, y_val = train_test_split(x_temp, y_temp, test_size=0.125,
                                                stratify=y_temp, random_state=seed)
    return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


class TextEncoder:
    """Turns raw strings into fixed-length sequences of word indices."""

    def __init__(self, texts, vocab=VOCAB, max_len=MAX_LEN):
        self.layer = layers.TextVectorization(max_tokens=vocab, output_sequence_length=max_len)
        self.layer.adapt(tf.constant(list(texts)))

    def __call__(self, texts):
        return self.layer(tf.constant(list(texts))).numpy()


def build_lstm(vocab=VOCAB, max_len=MAX_LEN, embed_dim=128, units=64):
    model = keras.Sequential([
        keras.Input(shape=(max_len,), dtype='int32'),
        layers.Embedding(vocab, embed_dim, mask_zero=True),   # mask_zero: LSTM ignores padding
        layers.LSTM(units),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def predict_texts(model, encoder, texts):
    """Probability that each raw review is positive."""
    return model.predict(encoder([clean(t) for t in texts]), verbose=0).ravel()


def show_examples(x_test, y_test, probs, n=10, seed=0):
    preds = (probs >= 0.5).astype(int)
    idx = np.random.default_rng(seed).choice(len(x_test), n, replace=False)
    table = pd.DataFrame({
        'Review (first 120 chars)': [x_test[i][:120] + '...' for i in idx],
        'True': ['positive' if y_test[i] else 'negative' for i in idx],
        'Predicted': ['positive' if preds[i] else 'negative' for i in idx],
        'P(positive)': probs[idx].round(3),
    })
    with pd.option_context('display.max_colwidth', 130):
        print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(range(n), probs[idx], color=['green' if preds[i] == y_test[i] else 'red' for i in idx])
    ax.axvline(0.5, color='k', ls='--')
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'#{i} (true: {"pos" if y_test[i] else "neg"})' for i in idx])
    ax.set_xlim(0, 1); ax.set_xlabel('P(positive)'); ax.set_title('Example predictions')
    plt.tight_layout(); plt.show()
    return table


def run(epochs=10):
    set_seed()
    (x_tr, y_tr), (x_val, y_val), (x_test, y_test) = load_imdb()
    print('Train:', len(x_tr), 'Val:', len(x_val), 'Test:', len(x_test))
    encoder = TextEncoder(x_tr)
    s_tr, s_val, s_test = encoder(x_tr), encoder(x_val), encoder(x_test)

    model = build_lstm()
    model.summary()
    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(s_tr, y_tr, validation_data=(s_val, y_val), epochs=epochs,
                        batch_size=128, callbacks=[early], verbose=2)
    plot_history(history, 'LSTM')

    test_loss, test_acc = model.evaluate(s_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}')
    probs = model.predict(s_test, verbose=0).ravel()
    from sklearn.metrics import classification_report
    print(classification_report(y_test, (probs >= 0.5).astype(int), target_names=['negative', 'positive'], digits=4))
    plot_confusion(y_test, (probs >= 0.5).astype(int), ['negative', 'positive'])
    show_examples(x_test, y_test, probs)

    mine = ['This movie was absolutely wonderful, the acting and story were brilliant.',
            'A complete waste of time. Boring plot and terrible acting.',
            'It was okay, not great but not bad either.']
    for text, p in zip(mine, predict_texts(model, encoder, mine)):
        print(f'{p:.3f}  {"POSITIVE" if p >= 0.5 else "NEGATIVE":8s}  {text}')
    return model, encoder
