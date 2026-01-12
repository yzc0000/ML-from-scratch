"""
Test CNN on MNIST Dataset (Handwritten Digits)

This script tests the cnn_model function on the classic MNIST dataset 
to verify the CNN integration into the dl_framework.

Architecture (Deeper CNN):
    Input (28x28x1) -> Conv(32, 3x3, pad=1) -> ReLU -> Conv(32, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
    -> Conv(64, 3x3, pad=1) -> ReLU -> Conv(64, 3x3, pad=1) -> ReLU -> MaxPool(2x2)
    -> Flatten -> Dense(128) -> ReLU -> Dense(64) -> ReLU -> Dense(10) -> Softmax
"""
import numpy as np
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_framework import cnn_model, cnn_predict, accuracy
from dl_framework.utils import update_lr


# Class names for MNIST
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def read_idx_images(filepath):
    """Read IDX format image file."""
    import struct
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images


def read_idx_labels(filepath):
    """Read IDX format label file."""
    import struct
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist():
    """Download and load MNIST dataset from Kaggle Hub."""
    print("Downloading MNIST from Kaggle Hub...")
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print(f"Dataset path: {path}")
    
    # Load IDX binary files
    X_train = read_idx_images(f"{path}/train-images.idx3-ubyte")
    Y_train_labels = read_idx_labels(f"{path}/train-labels.idx1-ubyte")
    X_test = read_idx_images(f"{path}/t10k-images.idx3-ubyte")
    Y_test_labels = read_idx_labels(f"{path}/t10k-labels.idx1-ubyte")
    
    # Reshape to (m, 28, 28, 1) for CNN and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype(np.float64) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype(np.float64) / 255.0
    
    # One-hot encode labels: shape (10, m)
    num_classes = 10
    Y_train_onehot = np.eye(num_classes)[Y_train_labels].T
    Y_test_onehot = np.eye(num_classes)[Y_test_labels].T
    
    print(f"Training set: X={X_train.shape}, Y={Y_train_onehot.shape}")
    print(f"Test set: X={X_test.shape}, Y={Y_test_onehot.shape}")
    
    return X_train, Y_train_onehot, X_test, Y_test_onehot, Y_test_labels


def test_cnn_simple():
    """Test a simple CNN on MNIST dataset."""
    print("=" * 60)
    print("CNN Framework Test - MNIST (Simple Architecture)")
    print("=" * 60)
    
    # Load data
    X_train, Y_train, X_test, Y_test, Y_test_labels = load_mnist()
    
    # Simple CNN layers (fast, good baseline)
    layers = [
        # Conv block 1: 28x28x1 -> 28x28x8 -> 14x14x8
        {"type": "conv", "filters": 8, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2, "mode": "max"},
        # Conv block 2: 14x14x8 -> 14x14x16 -> 7x7x16
        {"type": "conv", "filters": 16, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2, "mode": "max"},
        # Dense layers: 7*7*16 = 784 -> 64 -> 10
        {"type": "flatten"},
        {"type": "dense", "units": 64, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"}
    ]
    
    print("\nSimple Model Architecture:")
    print("  Input: 28x28x1")
    print("  Conv2D: 16 filters, 3x3, pad=1 -> 28x28x16")
    print("  MaxPool: 2x2 -> 14x14x16")
    print("  Conv2D: 32 filters, 3x3, pad=1 -> 14x14x32")
    print("  MaxPool: 2x2 -> 7x7x32")
    print("  Flatten: 1568")
    print("  Dense: 128, ReLU")
    print("  Dense: 10, Softmax")
    
    # Train
    print("\nTraining...")
    parameters, costs = cnn_model(
        X_train, Y_train, layers,
        num_epochs=15,
        mini_batch_size=64,
        learning_rate=0.001,
        print_cost=True,
        print_interval=1,
        plot_cost=False,
        lr_decay=update_lr,
        decay_rate=0.1,
        print_lr=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds, probs = cnn_predict(X_test, layers, parameters)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(10):
        pred_name = CLASS_NAMES[preds[i]]
        actual_name = CLASS_NAMES[Y_test_labels[i]]
        match = "✓" if preds[i] == Y_test_labels[i] else "✗"
        print(f"  {match} Predicted: {pred_name} | Actual: {actual_name}")
    
    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(f'CNN MNIST Training Cost (Test Accuracy: {acc:.2f}%)')
    plt.grid(True)
    plt.savefig('cnn_mnist_simple_training_cost.png', dpi=100)
    plt.close()
    print("\nSaved: cnn_mnist_simple_training_cost.png")
    
    return acc


def test_cnn_deep():
    """Test a deeper CNN on MNIST dataset (VGG-style)."""
    print("=" * 60)
    print("CNN Framework Test - MNIST (Deep VGG-Style Architecture)")
    print("=" * 60)
    
    # Load data
    X_train, Y_train, X_test, Y_test, Y_test_labels = load_mnist()
    
    # Deep CNN layers (VGG-style: double conv before pool)
    layers = [
        # Conv block 1: 28x28x1 -> 28x28x32 -> 28x28x32 -> 14x14x32
        {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "conv", "filters": 32, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2, "mode": "max"},
        # Conv block 2: 14x14x32 -> 14x14x64 -> 14x14x64 -> 7x7x64
        {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "conv", "filters": 64, "kernel_size": 3, "stride": 1, "pad": 1, "activation": "relu"},
        {"type": "pool", "pool_size": 2, "stride": 2, "mode": "max"},
        # Dense layers: 7*7*64 = 3136 -> 256 -> 128 -> 10
        {"type": "flatten"},
        {"type": "dense", "units": 256, "activation": "relu"},
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"}
    ]
    
    print("\nDeep VGG-Style Model Architecture:")
    print("  Input: 28x28x1")
    print("  Conv2D: 32 filters, 3x3, pad=1 -> 28x28x32")
    print("  Conv2D: 32 filters, 3x3, pad=1 -> 28x28x32")
    print("  MaxPool: 2x2 -> 14x14x32")
    print("  Conv2D: 64 filters, 3x3, pad=1 -> 14x14x64")
    print("  Conv2D: 64 filters, 3x3, pad=1 -> 14x14x64")
    print("  MaxPool: 2x2 -> 7x7x64")
    print("  Flatten: 3136")
    print("  Dense: 256, ReLU")
    print("  Dense: 128, ReLU")
    print("  Dense: 10, Softmax")
    
    # Train
    print("\nTraining Deep CNN (this will take longer)...")
    parameters, costs = cnn_model(
        X_train, Y_train, layers,
        num_epochs=20,
        mini_batch_size=64,
        learning_rate=0.001,
        print_cost=True,
        print_interval=1,
        plot_cost=False,
        lr_decay=update_lr,
        decay_rate=0.1,
        print_lr=True,
        keep_prob=0.9,  
        lambd=0.0001   
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds, probs = cnn_predict(X_test, layers, parameters)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(10):
        pred_name = CLASS_NAMES[preds[i]]
        actual_name = CLASS_NAMES[Y_test_labels[i]]
        match = "✓" if preds[i] == Y_test_labels[i] else "✗"
        print(f"  {match} Predicted: {pred_name} | Actual: {actual_name}")
    
    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(f'Deep CNN MNIST Training Cost (Test Accuracy: {acc:.2f}%)')
    plt.grid(True)
    plt.savefig('cnn_mnist_deep_training_cost.png', dpi=100)
    plt.close()
    print("\nSaved: cnn_mnist_deep_training_cost.png")
    
    return acc


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "deep":
        test_cnn_deep()
    else:
        test_cnn_simple()
