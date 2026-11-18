"""Practical 1: single-layer perceptron from scratch with NumPy (CO1).

Trains on AND / OR gates and a synthetic 2-D dataset, draws decision boundaries,
and shows that XOR cannot be learned.
"""
import matplotlib.pyplot as plt
import numpy as np

X_GATE = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
GATES = {
    'AND': np.array([0, 0, 0, 1]),
    'OR': np.array([0, 1, 1, 1]),
    'XOR': np.array([0, 1, 1, 0]),
}


def predict(x, w, b):
    """Step(w.x + b): 1 if the weighted sum is >= 0, else 0. Works on one row or many."""
    z = np.dot(x, w) + b
    return np.where(z >= 0, 1, 0)


class Perceptron:
    """Perceptron learning rule: w += lr * (target - prediction) * x, b += lr * (target - prediction)."""

    def __init__(self, learning_rate=0.1, epochs=50, verbose=False):
        self.lr = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_ = []                          # misclassifications in each epoch
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - int(predict(xi, self.w, self.b)))
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            self.errors_.append(errors)
            if self.verbose:
                print(epoch, self.w, self.b, errors)
            if errors == 0:                        # a full pass with no mistakes: converged
                break
        return self

    def predict(self, X):
        return predict(X, self.w, self.b)

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


def make_blobs(n_per_class=100, spread=0.8, seed=42):
    """Two Gaussian clusters around (-2, -2) and (2, 2): linearly separable."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(size=(n_per_class, 2)) * spread + [-2, -2],
                   rng.normal(size=(n_per_class, 2)) * spread + [2, 2]])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def plot_decision_boundary(model, X, y, title='', ax=None):
    """Shaded prediction regions, the data points and the line w1*x1 + w2*x2 + b = 0."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='coolwarm')

    zeros, ones = X[y == 0], X[y == 1]
    ax.scatter(zeros[:, 0], zeros[:, 1], c='tab:blue', edgecolor='k', label='Class 0')
    ax.scatter(ones[:, 0], ones[:, 1], c='tab:red', edgecolor='k', marker='s', label='Class 1')

    w1, w2 = model.w
    if w2 != 0:
        xs = np.array([x_min, x_max])
        ax.plot(xs, (-model.b - w1 * xs) / w2, 'k--', lw=2, label='Boundary')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title(title); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.legend(loc='best', fontsize=8)
    return ax


def run():
    # AND and OR gates
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name in zip(axes, ['AND', 'OR']):
        p = Perceptron(learning_rate=0.1, epochs=20).fit(X_GATE, GATES[name])
        print(f'{name}: w={p.w}, b={p.b:.2f}, predictions={p.predict(X_GATE)}, '
              f'target={GATES[name]}, epochs={len(p.errors_)}')
        plot_decision_boundary(p, X_GATE, GATES[name], f'{name} gate', ax)
    plt.tight_layout(); plt.show()

    # synthetic dataset, 80/20 split
    X, y = make_blobs()
    split = int(0.8 * len(X))
    p = Perceptron(learning_rate=0.01, epochs=50).fit(X[:split], y[:split])
    print(f'Synthetic: train acc {p.score(X[:split], y[:split]):.3f}, test acc {p.score(X[split:], y[split:]):.3f}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_decision_boundary(p, X, y, 'Synthetic dataset', axes[0])
    axes[1].plot(range(1, len(p.errors_) + 1), p.errors_, marker='o')
    axes[1].set_title('Misclassifications per epoch'); axes[1].set_xlabel('Epoch')
    plt.tight_layout(); plt.show()

    # XOR: not linearly separable, never converges
    p = Perceptron(learning_rate=0.1, epochs=50).fit(X_GATE, GATES['XOR'])
    print('XOR predictions:', p.predict(X_GATE), 'target:', GATES['XOR'],
          '| errors in last 5 epochs:', p.errors_[-5:])
    plot_decision_boundary(p, X_GATE, GATES['XOR'], 'XOR gate (cannot be separated)')
    plt.show()
    return p
