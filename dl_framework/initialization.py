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


def initialize_conv_parameters(filter_size, n_C_prev, n_C, method="he", seed=None):
    """
    Initialize parameters for a single convolutional layer.
    
    Arguments:
    filter_size -- integer, spatial size of filters (f x f)
    n_C_prev -- integer, number of input channels
    n_C -- integer, number of output filters
    method -- initialization method ("he", "xavier", "random")
    seed -- optional random seed
    
    Returns:
    W -- weights of shape (f, f, n_C_prev, n_C)
    b -- biases of shape (1, 1, 1, n_C)
    """
    if seed is not None:
        np.random.seed(seed)
    
    f = filter_size
    fan_in = f * f * n_C_prev
    
    if method == "he":
        W = np.random.randn(f, f, n_C_prev, n_C) * np.sqrt(2 / fan_in)
    elif method == "xavier":
        W = np.random.randn(f, f, n_C_prev, n_C) * np.sqrt(1 / fan_in)
    else:  # random
        W = np.random.randn(f, f, n_C_prev, n_C) * 0.01
    
    b = np.zeros((1, 1, 1, n_C))
    
    return W, b


def initialize_cnn_parameters(layers, input_shape, seed=1):
    """
    Initialize parameters for a CNN model based on layer configurations.
    
    Arguments:
    layers -- list of layer configuration dicts
    input_shape -- tuple (n_H, n_W, n_C) for input images
    seed -- random seed
    
    Returns:
    parameters -- dict containing all initialized W and b
    """
    np.random.seed(seed)
    parameters = {}
    current_shape = input_shape
    conv_idx = 0
    dense_idx = 0
    current_units = None
    
    for i, layer in enumerate(layers):
        if layer["type"] == "conv":
            conv_idx += 1
            n_C_prev = current_shape[2]
            n_C = layer["filters"]
            f = layer["kernel_size"]
            
            W, b = initialize_conv_parameters(f, n_C_prev, n_C, method="he")
            parameters[f"W_conv{conv_idx}"] = W
            parameters[f"b_conv{conv_idx}"] = b
            
            # Update shape after conv
            n_H = int((current_shape[0] - f + 2 * layer["pad"]) / layer["stride"]) + 1
            n_W = int((current_shape[1] - f + 2 * layer["pad"]) / layer["stride"]) + 1
            current_shape = (n_H, n_W, n_C)
            
        elif layer["type"] == "pool":
            f = layer["pool_size"]
            stride = layer["stride"]
            n_H = int((current_shape[0] - f) / stride) + 1
            n_W = int((current_shape[1] - f) / stride) + 1
            current_shape = (n_H, n_W, current_shape[2])
            
        elif layer["type"] == "flatten":
            current_units = current_shape[0] * current_shape[1] * current_shape[2]
            
        elif layer["type"] == "dense":
            dense_idx += 1
            units = layer["units"]
            
            # He initialization for dense layers
            W = np.random.randn(units, current_units) * np.sqrt(2 / current_units)
            b = np.zeros((units, 1))
            
            parameters[f"W_dense{dense_idx}"] = W
            parameters[f"b_dense{dense_idx}"] = b
            
            current_units = units
        
        elif layer["type"] == "residual":
            res_idx = sum(1 for l in layers[:i+1] if l["type"] == "residual")
            n_C_prev = current_shape[2]
            n_C = layer["filters"]
            f = layer["kernel_size"]
            downsample = layer["downsample"]
            
            # Conv1: first conv in residual block
            W1, b1 = initialize_conv_parameters(f, n_C_prev, n_C, method="he")
            parameters[f"W_res{res_idx}_conv1"] = W1
            parameters[f"b_res{res_idx}_conv1"] = b1
            
            # Conv2: second conv in residual block
            W2, b2 = initialize_conv_parameters(f, n_C, n_C, method="he")
            parameters[f"W_res{res_idx}_conv2"] = W2
            parameters[f"b_res{res_idx}_conv2"] = b2
            
            # Skip connection: 1x1 conv if dimensions change
            if n_C_prev != n_C or downsample:
                W_skip, b_skip = initialize_conv_parameters(1, n_C_prev, n_C, method="he")
                parameters[f"W_res{res_idx}_skip"] = W_skip
                parameters[f"b_res{res_idx}_skip"] = b_skip
                layer["has_skip_conv"] = True
            else:
                layer["has_skip_conv"] = False
            
            # Update shape
            if downsample:
                n_H = current_shape[0] // 2
                n_W = current_shape[1] // 2
            else:
                n_H = current_shape[0]
                n_W = current_shape[1]
            current_shape = (n_H, n_W, n_C)
    
    return parameters





