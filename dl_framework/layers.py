"""
Deep Learning Framework - Forward/Backward Propagation Layers
"""
import numpy as np

from .activations import sigmoid, relu, softmax, sigmoid_backward, relu_backward
from .regularization import batchnorm_forward, batchnorm_backward


def linear_forward(A, W, b):
    """Compute Z = W @ A + b."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_backward(dZ, cache, lambd=0, W=None):
    """Backward propagation for linear layer with optional L2 regularization."""
    A_prev, W_cache, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    if lambd > 0:
        dW += (lambd / m) * W_cache
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W_cache.T, dZ)
    
    return dA_prev, dW, db


def L_model_forward(X, parameters, output="sigmoid", keep_prob=1.0, 
                    bn_params=None, bn_running=None, training=True):
    """
    L-layer forward propagation with configurable options.
    """
    caches = []
    A = X
    L = len(parameters) // 2
    use_dropout = keep_prob < 1.0 and training  # Only dropout during training
    use_batchnorm = bn_params is not None
    
    if use_dropout:
        np.random.seed(1)
    
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        # Linear
        Z, linear_cache = linear_forward(A_prev, W, b)
        
        # Batch Normalization (optional)
        if use_batchnorm:
            gamma = bn_params['gamma' + str(l)]
            beta = bn_params['beta' + str(l)]
            Z, bn_cache = batchnorm_forward(
                Z, gamma, beta, 
                bn_running=bn_running, 
                layer_idx=l, 
                training=training
            )
        else:
            bn_cache = None
        
        A, activation_cache = relu(Z)
        
        if use_dropout:
            D = np.random.rand(*A.shape)
            D = (D < keep_prob).astype(int)
            A = A * D / keep_prob
        else:
            D = None
        
        cache = (linear_cache, activation_cache, bn_cache, D)
        caches.append(cache)
    
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z, linear_cache = linear_forward(A, W, b)
    
    if output == "softmax":
        AL, activation_cache = softmax(Z)
    else:  # sigmoid
        AL, activation_cache = sigmoid(Z)
    
    cache = (linear_cache, activation_cache, None, None)
    caches.append(cache)
    
    return AL, caches


def L_model_backward(AL, Y, caches, parameters, output="sigmoid", keep_prob=1.0, 
                     lambd=0, bn_params=None):
    """
    L-layer backward propagation with configurable options.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    use_dropout = keep_prob < 1.0
    use_batchnorm = bn_params is not None
    
    # Output layer gradient
    linear_cache, activation_cache, _, _ = caches[L - 1]
    
    epsilon = 1e-15  # Prevent divide by zero
    if output == "softmax":
        dZ = AL - Y
    else:  # sigmoid
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        dAL = -(np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped))
        dZ = sigmoid_backward(dAL, activation_cache)
    
    A_prev, W, b = linear_cache
    grads["dW" + str(L)] = 1/m * np.dot(dZ, A_prev.T)
    if lambd > 0:
        grads["dW" + str(L)] += (lambd / m) * parameters['W' + str(L)]
    grads["db" + str(L)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dA" + str(L-1)] = np.dot(W.T, dZ)
    
    # Hidden layers (reverse order)
    for l in reversed(range(L - 1)):
        linear_cache, activation_cache, bn_cache, D = caches[l]
        
        dA = grads["dA" + str(l + 1)]
        
        # Dropout backward
        if use_dropout and D is not None:
            dA = dA * D / keep_prob
        
        # ReLU backward
        dZ = relu_backward(dA, activation_cache)
        
        # Batch norm backward
        if use_batchnorm and bn_cache is not None:
            dZ, dgamma, dbeta = batchnorm_backward(dZ, bn_cache)
            grads["dgamma" + str(l + 1)] = dgamma
            grads["dbeta" + str(l + 1)] = dbeta
        
        # Linear backward with optional L2
        A_prev, W, b = linear_cache
        grads["dW" + str(l + 1)] = 1/m * np.dot(dZ, A_prev.T)
        if lambd > 0:
            grads["dW" + str(l + 1)] += (lambd / m) * parameters['W' + str(l + 1)]
        grads["db" + str(l + 1)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l)] = np.dot(W.T, dZ)
    
    return grads
