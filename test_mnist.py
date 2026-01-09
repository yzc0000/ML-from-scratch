"""
Test DL Framework on MNIST Dataset (from Kaggle Hub)

This script downloads MNIST from Kaggle and tests our framework on 
handwritten digit classification (784 inputs -> 10 classes).
"""
import numpy as np
import pandas as pd
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_framework import (
    model, predict, accuracy,
    train_test_split, train_val_test_split, save_model,
    update_lr  
)


def read_idx_images(filepath):
    """Read IDX format image file."""
    import struct
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    return images


def read_idx_labels(filepath):
    """Read IDX format label file."""
    import struct
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist_from_kaggle():
    """Download and load MNIST dataset from Kaggle Hub."""
    print("Downloading MNIST from Kaggle Hub...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print(f"Dataset path: {path}")
    
    import os
    print(f"Files: {os.listdir(path)}")
    
    # Load IDX binary files
    X_train = read_idx_images(f"{path}/train-images.idx3-ubyte")
    Y_train_labels = read_idx_labels(f"{path}/train-labels.idx1-ubyte")
    X_test = read_idx_images(f"{path}/t10k-images.idx3-ubyte")
    Y_test_labels = read_idx_labels(f"{path}/t10k-labels.idx1-ubyte")
    
    # Transpose to (features, samples) and normalize to [0, 1]
    X_train = X_train.T / 255.0
    X_test = X_test.T / 255.0
    
    # One-hot encode labels for softmax
    num_classes = 10
    Y_train_onehot = np.eye(num_classes)[Y_train_labels].T
    Y_test_onehot = np.eye(num_classes)[Y_test_labels].T
    
    print(f"Training set: X={X_train.shape}, Y={Y_train_onehot.shape}")
    print(f"Test set: X={X_test.shape}, Y={Y_test_onehot.shape}")
    
    return X_train, Y_train_onehot, X_test, Y_test_onehot, Y_test_labels


def visualize_samples(X, Y_labels, num_samples=10):
    """Visualize some sample images."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = X[:, i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Label: {Y_labels[i]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=100)
    plt.close()
    print("Saved: mnist_samples.png")


def test_mnist():
    """Test framework on MNIST dataset."""
    print("=" * 60)
    print("DL Framework - MNIST Test (784 -> 10 classes)")
    print("=" * 60)
    
    # Load data
    X_train, Y_train, X_test, Y_test, Y_test_labels = load_mnist_from_kaggle()
    
    # Visualize samples
    visualize_samples(X_train, np.argmax(Y_train, axis=0))

    
    # Define network architecture
    # 784 inputs -> 256 -> 128 -> 10 outputs
    layers_dims = [784, 256, 128,64,32, 10]
    
    print(f"\nNetwork architecture: {layers_dims}")
    print("Training...")
    
    # Train the model
    params, bn_params, bn_running, costs = model(
        X_train, Y_train, layers_dims,
        output="softmax",
        optimizer="adam",
        learning_rate=0.001,
        mini_batch_size=128,
        num_epochs=20,
        lambd=0.001,
        use_batchnorm=True,
        lr_decay=update_lr,  
        decay_rate=1,      
        init_method="he",
        print_cost=True,
        print_interval=1,
        plot_cost=True,
        print_lr=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds, probs = predict(X_test, params, output="softmax", bn_params=bn_params, bn_running=bn_running)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # Show some predictions
    print("\nSample predictions vs actual:")
    for i in range(10):
        print(f"  Predicted: {preds[i]}, Actual: {Y_test_labels[i]}")
    
    # Save model
    save_model('mnist_model.npz', params)
    
    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iterations (per 100)')
    plt.ylabel('Cost')
    plt.title(f'MNIST Training Cost (Final Accuracy: {acc:.2f}%)')
    plt.grid(True)
    plt.savefig('mnist_training_cost.png', dpi=100)
    plt.close()
    print("Saved: mnist_training_cost.png")
    
    return acc


if __name__ == "__main__":
    test_mnist()
