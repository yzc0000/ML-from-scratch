"""
Deep Learning Framework - Activation Functions
"""
import numpy as np


def sigmoid(Z):
    """
    Compute sigmoid activation with numerical stability.
    Uses clipping to prevent overflow in exp.
    """
    # Clip Z to prevent overflow: sigmoid(-709) ≈ 0, sigmoid(709) ≈ 1
    Z_clipped = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z_clipped))
    cache = Z
    return A, cache


def relu(Z):
    """Compute ReLU activation."""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def softmax(Z):
    """
    Compute softmax activation for multi-class classification.
    awdawda
    Arguments:
    Z -- pre-activation values, shape (num_classes, m)
    
    Returns:
    A -- softmax output (probabilities), shape (num_classes, m)
    cache -- Z for backward pass
    """
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache = Z
    return A, cache


def tanh_activation(Z):
    """Compute tanh activation."""
    A = np.tanh(Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """Backward propagation for sigmoid with numerical stability."""
    Z = cache
    # Clip Z to prevent overflow
    Z_clipped = np.clip(Z, -500, 500)
    s = 1 / (1 + np.exp(-Z_clipped))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """Backward propagation for ReLU."""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax_backward(dA, cache):
    """
    Backward propagation for softmax.
    Note: When used with categorical cross-entropy, dZ = A - Y directly.
    """
    Z = cache
    A, _ = softmax(Z)
    dZ = A * (dA - np.sum(dA * A, axis=0, keepdims=True))
    return dZ


def tanh_backward(dA, cache):
    """Backward propagation for tanh."""
    Z = cache
    A = np.tanh(Z)
    dZ = dA * (1 - A**2)
    return dZ
