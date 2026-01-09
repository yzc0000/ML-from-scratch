"""
Deep Learning Framework - Regularization (Batch Norm, Dropout, L2)
"""
import numpy as np

from .activations import relu, relu_backward


def initialize_bn_running_stats(layers_dims):
    """
    Initialize running statistics for batch normalization inference.
    Note: BN is NOT applied to the output layer.
    """
    bn_running = {}
    L = len(layers_dims)
    
    # Only initialize for hidden layers (1 to L-1), NOT the output layer
    for l in range(1, L - 1):
        bn_running['running_mean' + str(l)] = np.zeros((layers_dims[l], 1))
        bn_running['running_var' + str(l)] = np.ones((layers_dims[l], 1))
    
    return bn_running


def batchnorm_forward(Z, gamma, beta, bn_running=None, layer_idx=None, 
                       training=True, momentum=0.9, epsilon=1e-8):
    if training:
        # Training mode: use batch statistics
        mu = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        
        # Update running statistics if provided
        if bn_running is not None and layer_idx is not None:
            key_mean = 'running_mean' + str(layer_idx)
            key_var = 'running_var' + str(layer_idx)
            bn_running[key_mean] = momentum * bn_running[key_mean] + (1 - momentum) * mu
            bn_running[key_var] = momentum * bn_running[key_var] + (1 - momentum) * var
    else:
        # Inference mode: use running statistics
        if bn_running is None or layer_idx is None:
            raise ValueError("bn_running and layer_idx required for inference mode")
        mu = bn_running['running_mean' + str(layer_idx)]
        var = bn_running['running_var' + str(layer_idx)]
    
    # Normalize
    Z_centered = Z - mu
    Z_hat = Z_centered / np.sqrt(var + epsilon)
    
    # Scale and shift
    Z_norm = gamma * Z_hat + beta
    
    cache = (Z, Z_hat, Z_centered, mu, var, gamma, beta, epsilon)
    return Z_norm, cache


def batchnorm_backward(dZ_norm, cache):
    """Batch normalization backward pass."""
    Z, Z_hat, Z_centered, mu, var, gamma, beta, epsilon = cache
    m = Z.shape[1]
    
    dgamma = np.sum(dZ_norm * Z_hat, axis=1, keepdims=True)
    dbeta = np.sum(dZ_norm, axis=1, keepdims=True)
    
    dZ_hat = dZ_norm * gamma
    std_inv = 1 / np.sqrt(var + epsilon)
    dvar = np.sum(dZ_hat * Z_centered * (-0.5) * (var + epsilon)**(-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_hat * (-std_inv), axis=1, keepdims=True) + dvar * np.mean(-2 * Z_centered, axis=1, keepdims=True)
    dZ = dZ_hat * std_inv + dvar * 2 * Z_centered / m + dmu / m
    
    return dZ, dgamma, dbeta


def update_bn_parameters(bn_params, grads, learning_rate):
    """Update batch normalization parameters (gamma, beta)."""
    L = len(bn_params) // 2

    for l in range(1, L + 1):
        if 'dgamma' + str(l) in grads:
            bn_params['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
            bn_params['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]

    return bn_params


def dropout_forward(A, keep_prob):
    D = np.random.rand(*A.shape)
    D = (D < keep_prob).astype(int)
    A = A * D / keep_prob
    return A, D


def dropout_backward(dA, D, keep_prob):
    """Apply dropout mask during backpropagation."""
    dA = dA * D / keep_prob
    return dA
