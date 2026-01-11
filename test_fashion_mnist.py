"""
Test DL Framework on Fashion-MNIST Dataset (from Kaggle Hub)

Fashion-MNIST: 70,000 grayscale images of 10 clothing categories.
Same structure as MNIST (28x28) but more challenging.

Classes:
0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat
5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot

Result:
Achieved 90.28% accuracy on test set with a relatively small neural network and number of epochs.
"""
import numpy as np
import kagglehub
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_framework import (
    model, predict, accuracy,
    train_test_split, save_model,
    update_lr
)

# Class names for Fashion-MNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


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


def load_fashion_mnist():
    """Download and load Fashion-MNIST dataset from Kaggle Hub."""
    print("Downloading Fashion-MNIST from Kaggle Hub...")
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    print(f"Dataset path: {path}")
    
    import os
    print(f"Files: {os.listdir(path)}")
    
    # Load IDX binary files (same format as MNIST)
    X_train = read_idx_images(f"{path}/train-images-idx3-ubyte")
    Y_train_labels = read_idx_labels(f"{path}/train-labels-idx1-ubyte")
    X_test = read_idx_images(f"{path}/t10k-images-idx3-ubyte")
    Y_test_labels = read_idx_labels(f"{path}/t10k-labels-idx1-ubyte")
    
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
    """Visualize some sample images with class names."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = X[:, i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{CLASS_NAMES[Y_labels[i]]}")
            ax.axis('off')
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png', dpi=100)
    plt.close()
    print("Saved: fashion_mnist_samples.png")


def test_fashion_mnist():
    """Test framework on Fashion-MNIST dataset."""
    print("=" * 60)
    print("DL Framework - Fashion-MNIST Test (784 -> 10 classes)")
    print("=" * 60)
    
    # Load data
    X_train, Y_train, X_test, Y_test, Y_test_labels = load_fashion_mnist()
    
    # Visualize samples
    visualize_samples(X_train, np.argmax(Y_train, axis=0))
    
    # Define network architecture 
    layers_dims = [784, 512, 256, 128, 64, 10]
    
    print(f"\nNetwork architecture: {layers_dims}")
    
    # Train the model
    params, bn_params, bn_running, costs = model(
        X_train, Y_train, layers_dims,
        output="softmax",
        optimizer="adam",
        learning_rate=0.001,
        mini_batch_size=128,
        num_epochs=30,  
        lambd=0.001,
        use_batchnorm=True,
        lr_decay=update_lr,
        decay_rate=0.25, 
        init_method="he",
        print_cost=True,
        print_interval=1,
        print_lr=True,
        plot_cost=True,
        keep_prob = 0.81    
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds, probs = predict(X_test, params, output="softmax", 
                           bn_params=bn_params, bn_running=bn_running)
    acc = accuracy(preds, Y_test)
    print(f"\nTest Accuracy: {acc:.2f}%")
    
    # Show some predictions
    print("\nSample predictions vs actual:")
    for i in range(10):
        pred_name = CLASS_NAMES[preds[i]]
        actual_name = CLASS_NAMES[Y_test_labels[i]]
        match = "✓" if preds[i] == Y_test_labels[i] else "✗"
        print(f"  {match} Predicted: {pred_name:12} | Actual: {actual_name}")
    
    # Save model
    save_model('fashion_mnist_model.npz', params)
    
    # Plot training cost
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(f'Fashion-MNIST Training Cost (Final Accuracy: {acc:.2f}%)')
    plt.grid(True)
    plt.savefig('fashion_mnist_training_cost.png', dpi=100)
    plt.close()
    print("Saved: fashion_mnist_training_cost.png")
    
    return acc


if __name__ == "__main__":
    test_fashion_mnist()
