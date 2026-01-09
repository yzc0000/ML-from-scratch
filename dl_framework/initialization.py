"""
Deep Learning Framework - Parameter Initialization
"""
import numpy as np


def initialize_parameters(layers_dims, method="he", seed=3):
    """
    Initialize parameters for an L-layer neural network.
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        if method == "zeros":
            parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        elif method == "random":
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        elif method == "xavier":
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(1 / layers_dims[l-1])
        else:  # "he" (default)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_bn_parameters(layers_dims):
    """
    Initialize batch normalization parameters (gamma and beta) for each hidden layer.
    Note: BN is NOT applied to the output layer.
    """
    bn_params = {}
    L = len(layers_dims)
    
    # Only initialize for hidden layers (1 to L-1), NOT the output layer
    for l in range(1, L - 1):
        bn_params['gamma' + str(l)] = np.ones((layers_dims[l], 1))
        bn_params['beta' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return bn_params
