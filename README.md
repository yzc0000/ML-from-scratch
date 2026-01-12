# Deep Learning Framework from Scratch

A fully-functional deep learning framework built from scratch using only NumPy. No TensorFlow, no PyTorch, just pure Python and linear algebra.

*Built as a learning project to understand deep learning fundamentals from first principles.*
## 🎯 Test Results

### Fully Connected Networks

| Dataset | Accuracy | Architecture |
|---------|----------|--------------|
| **MNIST** | 98.54% | [784, 256, 128, 64, 32, 10] |
| **Fashion-MNIST** | 90.56% | [784, 512, 256, 128, 64, 10] |

### Convolutional Neural Networks (CNN)

| Dataset | Accuracy | Architecture |
|---------|----------|--------------|
| **MNIST** | 98.92% | Conv(8) → Pool → Conv(16) → Pool → Dense(64) → Dense(10) |
| **Fashion-MNIST** | 91.41% | Conv(8) → Pool → Conv(16) → Pool → Dense(64) → Dense(10) |


## ✨ Features

### Core Components
- **Forward Propagation** - L-layer neural networks
- **Backpropagation** - Automatic gradient computation
- **Multi-class Classification** - Softmax + Cross-entropy loss
- **Binary Classification** - Sigmoid + Binary cross-entropy

### CNN Components
- **Convolutional Layers** - Vectorized im2col implementation
- **Pooling Layers** - Max pooling and average pooling
- **Residual Blocks** - ResNet-style skip connections
- **Flatten Layer** - Automatic shape inference

### Optimizers
- Gradient Descent (SGD)
- Momentum
- **Adam** (recommended)

### Regularization
- **L2 Regularization** (weight decay)
- **Dropout**
- **Batch Normalization**

### Training Features
- Mini-batch training
- **Learning rate decay** (inverse time decay)
- Configurable print intervals
- He/Xavier/Random initialization
- Model save/load functionality

## 📁 Project Structure

```
ML from scratch/
├── dl_framework/
│   ├── __init__.py          # Main exports
│   ├── activations.py       # ReLU, Sigmoid, Softmax, Tanh
│   ├── initialization.py    # He, Xavier, Random, Zero init (FC + CNN)
│   ├── layers.py            # Forward/Backward propagation (FC + CNN)
│   ├── losses.py            # Cross-entropy, BCE
│   ├── models.py            # Training loops (model, cnn_model)
│   ├── optimizers.py        # SGD, Momentum, Adam
│   ├── regularization.py    # BatchNorm, Dropout
│   └── utils.py             # Mini-batches, LR decay, im2col, save/load
├── test_mnist.py            # MNIST FC network test
├── test_fashion_mnist.py    # Fashion-MNIST FC network test
├── test_cnn_mnist.py        # MNIST CNN test
├── test_cnn_fashion_mnist.py # Fashion-MNIST CNN test (+ ResNet)
└── generate_data.py         # Synthetic datasets (circles, moons, spiral)
```

## 🚀 Quick Start

### Installation

```bash
pip install numpy matplotlib kagglehub
```

### Basic Usage

```python
from dl_framework import model, predict, accuracy

# Define architecture: [input, hidden1, hidden2, ..., output]
layers_dims = [784, 256, 128, 10]

# Train the model
params, bn_params, bn_running, costs = model(
    X_train, Y_train, layers_dims,
    output="softmax",           # "softmax" for multi-class, "sigmoid" for binary
    optimizer="adam",
    learning_rate=0.001,
    mini_batch_size=128,
    num_epochs=30,
    lambd=0.001,                # L2 regularization
    use_batchnorm=True,
    print_cost=True,
    print_interval=1
)

# Make predictions
predictions, probabilities = predict(
    X_test, params, 
    output="softmax",
    bn_params=bn_params, 
    bn_running=bn_running
)

# Evaluate
acc = accuracy(predictions, Y_test)
print(f"Test Accuracy: {acc:.2f}%")
```

### With Learning Rate Decay

```python
from dl_framework import model, update_lr

params, _, _, costs = model(
    X_train, Y_train, layers_dims,
    optimizer="adam",
    learning_rate=0.001,
    lr_decay=update_lr,    # Inverse time decay
    decay_rate=0.5,        # lr = lr0 / (1 + decay_rate * epoch)
    num_epochs=50
)
```

## 🧪 Running Tests

### Fully Connected Networks
```bash
# MNIST (Handwritten Digits)
python test_mnist.py

# Fashion-MNIST (Clothing Classification)
python test_fashion_mnist.py
```

### Convolutional Neural Networks
```bash
# MNIST CNN (Simple)
python test_cnn_mnist.py

# MNIST CNN (Deep VGG-style)
python test_cnn_mnist.py deep

# Fashion-MNIST CNN
python test_cnn_fashion_mnist.py

# Fashion-MNIST ResNet
python test_cnn_fashion_mnist.py resnet
```

### Synthetic Datasets
```bash
python generate_data.py
```

## 📊 Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output` | str | "sigmoid" | "sigmoid" or "softmax" |
| `optimizer` | str | "adam" | "gd", "momentum", or "adam" |
| `learning_rate` | float | 0.001 | Initial learning rate |
| `mini_batch_size` | int | 64 | Batch size for training |
| `num_epochs` | int | 5000 | Number of training epochs |
| `lambd` | float | 0 | L2 regularization strength |
| `keep_prob` | float | 1.0 | Dropout keep probability |
| `use_batchnorm` | bool | False | Enable batch normalization |
| `lr_decay` | func | None | Learning rate decay function |
| `decay_rate` | float | 0 | Decay rate for lr_decay |
| `init_method` | str | "he" | "he", "xavier", "random", "zeros" |
| `print_interval` | int | 100 | Print cost every N epochs |
| `print_lr` | bool | True | Print learning rate (when using lr_decay) |

---


