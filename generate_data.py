"""
Generate synthetic datasets for testing the DL Framework.
Creates classification datasets with thousands of data points.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_framework import (
    model, predict, accuracy,
    train_test_split, save_model
)


def generate_circles(n_samples=5000, noise=0.1, seed=42):
    """
    Generate two concentric circles (non-linear classification).
    
    Returns:
    X -- features, shape (2, n_samples)
    Y -- labels, shape (1, n_samples)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Inner circle (class 0)
    theta_inner = np.random.uniform(0, 2*np.pi, n_per_class)
    r_inner = 0.3 + np.random.randn(n_per_class) * noise
    X_inner = np.vstack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    
    # Outer circle (class 1)
    theta_outer = np.random.uniform(0, 2*np.pi, n_per_class)
    r_outer = 0.8 + np.random.randn(n_per_class) * noise
    X_outer = np.vstack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
    
    X = np.hstack([X_inner, X_outer])
    Y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).reshape(1, -1)
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X = X[:, perm]
    Y = Y[:, perm]
    
    return X, Y.astype(int)


def generate_moons(n_samples=5000, noise=0.1, seed=42):
    """
    Generate two interleaving half circles.
    
    Returns:
    X -- features, shape (2, n_samples)
    Y -- labels, shape (1, n_samples)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # First moon
    theta1 = np.linspace(0, np.pi, n_per_class)
    X1 = np.vstack([np.cos(theta1), np.sin(theta1)])
    X1 += np.random.randn(2, n_per_class) * noise
    
    # Second moon (shifted and flipped)
    theta2 = np.linspace(0, np.pi, n_per_class)
    X2 = np.vstack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])
    X2 += np.random.randn(2, n_per_class) * noise
    
    X = np.hstack([X1, X2])
    Y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)]).reshape(1, -1)
    
    perm = np.random.permutation(n_samples)
    return X[:, perm], Y[:, perm].astype(int)


def generate_spiral(n_samples=6000, n_classes=3, noise=0.2, seed=42):
    """
    Generate spiral dataset for multi-class classification.
    
    Returns:
    X -- features, shape (2, n_samples)
    Y -- one-hot labels, shape (n_classes, n_samples)
    """
    np.random.seed(seed)
    n_per_class = n_samples // n_classes
    
    X = np.zeros((2, n_samples))
    labels = np.zeros(n_samples, dtype=int)
    
    for c in range(n_classes):
        idx = range(c * n_per_class, (c + 1) * n_per_class)
        r = np.linspace(0.1, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + np.random.randn(n_per_class) * noise
        X[0, idx] = r * np.sin(t)
        X[1, idx] = r * np.cos(t)
        labels[list(idx)] = c
    
    # One-hot encode
    Y = np.eye(n_classes)[labels].T
    
    perm = np.random.permutation(n_samples)
    return X[:, perm], Y[:, perm]


def generate_linear(n_samples=5000, n_features=10, seed=42):
    """
    Generate linearly separable dataset.
    
    Returns:
    X -- features, shape (n_features, n_samples)
    Y -- labels, shape (1, n_samples)
    """
    np.random.seed(seed)
    X = np.random.randn(n_features, n_samples)
    
    # Create a random hyperplane
    w = np.random.randn(n_features, 1)
    Y = (X.T @ w > 0).T.astype(int)
    
    return X, Y


def visualize_2d_data(X, Y, title="Dataset"):
    """Visualize 2D dataset."""
    plt.figure(figsize=(8, 6))
    
    if Y.shape[0] > 1:  # Multi-class (one-hot)
        labels = np.argmax(Y, axis=0)
    else:
        labels = Y.flatten()
    
    scatter = plt.scatter(X[0], X[1], c=labels, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter)
    plt.title(f"{title} ({X.shape[1]} samples)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=100)
    plt.close()
    print(f"Saved: {title.lower().replace(' ', '_')}.png")


def test_on_circles():
    """Test framework on circles dataset."""
    print("\n" + "="*50)
    print("Testing on CIRCLES dataset (5000 samples)")
    print("="*50)
    
    X, Y = generate_circles(n_samples=5000)
    visualize_2d_data(X, Y, "Circles Dataset")
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2)
    
    params, _, _, costs = model(
        X_train, Y_train, [2, 16, 8, 1],
        output="sigmoid",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=3000,
        print_cost=True,
        plot_cost=False
    )
    
    preds, _ = predict(X_test, params)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    save_model('circles_model.npz', params)
    return acc


def test_on_moons():
    """Test framework on moons dataset."""
    print("\n" + "="*50)
    print("Testing on MOONS dataset (5000 samples)")
    print("="*50)
    
    X, Y = generate_moons(n_samples=5000)
    visualize_2d_data(X, Y, "Moons Dataset")
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2)
    
    params, _, _, costs = model(
        X_train, Y_train, [2, 16, 8, 1],
        output="sigmoid",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=3000,
        print_cost=True,
        plot_cost=False
    )
    
    preds, _ = predict(X_test, params)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    save_model('moons_model.npz', params)
    return acc


def test_on_spiral():
    """Test framework on spiral dataset (multi-class)."""
    print("\n" + "="*50)
    print("Testing on SPIRAL dataset (6000 samples, 3 classes)")
    print("="*50)
    
    X, Y = generate_spiral(n_samples=6000, n_classes=3)
    visualize_2d_data(X, Y, "Spiral Dataset")
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2)
    
    params, _, _, costs = model(
        X_train, Y_train, [2, 32, 16, 3],
        output="softmax",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=5000,
        print_cost=True,
        plot_cost=False
    )
    
    preds, _ = predict(X_test, params, output="softmax")
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    save_model('spiral_model.npz', params)
    return acc


def test_on_large_linear():
    """Test on large linearly separable dataset."""
    print("\n" + "="*50)
    print("Testing on LARGE LINEAR dataset (10000 samples, 20 features)")
    print("="*50)
    
    X, Y = generate_linear(n_samples=10000, n_features=20)
    
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2)
    
    params, _, _, costs = model(
        X_train, Y_train, [20, 32, 16, 8, 1],
        output="sigmoid",
        optimizer="adam",
        learning_rate=0.001,
        num_epochs=2000,
        print_cost=True,
        plot_cost=False
    )
    
    preds, _ = predict(X_test, params)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    save_model('large_linear_model.npz', params)
    return acc


if __name__ == "__main__":
    print("="*50)
    print("DL Framework - Large Scale Testing")
    print("="*50)
    
    results = {}
    
    results['circles'] = test_on_circles()
    results['moons'] = test_on_moons()
    results['spiral'] = test_on_spiral()
    results['large_linear'] = test_on_large_linear()
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, acc in results.items():
        print(f"{name:15}: {acc:.2f}%")
