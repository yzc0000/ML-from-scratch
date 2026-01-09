import numpy as np
import copy
import matplotlib.pyplot as plt

np.random.seed(1)


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A"; stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize parameters for a 2-layer neural network.
    
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing W1, b1, W2, b2
    """
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for an L-layer neural network.
    
    Arguments:
    layer_dims -- python list containing the dimensions of each layer
    
    Returns:
    parameters -- python dictionary containing W1, b1, ..., WL, bL
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer: (size of previous layer, number of examples)
    W -- weights matrix: (size of current layer, size of previous layer)
    b -- bias vector: (size of current layer, 1)

    Returns:
    Z -- the pre-activation parameter
    cache -- tuple containing A, W, b for backprop
    """
    Z = W @ A + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement forward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    A_prev -- activations from previous layer
    W -- weights matrix
    b -- bias vector
    activation -- "sigmoid" or "relu"

    Returns:
    A -- the post-activation value
    cache -- tuple containing linear_cache and activation_cache
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data: (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output layer
    caches -- list of caches for each layer
    """
    caches = []
    A = X
    L = len(parameters) // 2

    # Hidden layers with ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev,
            parameters["W" + str(l)],
            parameters["b" + str(l)],
            activation="relu"
        )
        caches.append(cache)

    # Output layer with sigmoid
    AL, cache = linear_activation_forward(
        A,
        parameters["W" + str(L)],
        parameters["b" + str(L)],
        activation="sigmoid"
    )
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cross-entropy cost function.

    Arguments:
    AL -- probability predictions: (1, number of examples)
    Y -- true labels: (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer.

    Arguments:
    dZ -- Gradient of cost with respect to Z
    cache -- tuple (A_prev, W, b) from forward propagation

    Returns:
    dA_prev -- Gradient with respect to A_prev
    dW -- Gradient with respect to W
    db -- Gradient with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * dZ @ A_prev.T
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient
    cache -- tuple (linear_cache, activation_cache)
    activation -- "sigmoid" or "relu"
    
    Returns:
    dA_prev, dW, db
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    Implement backward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    AL -- probability vector from forward propagation
    Y -- true label vector
    caches -- list of caches from forward propagation
    
    Returns:
    grads -- dictionary with gradients
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    # Derivative of cost with respect to AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Output layer (sigmoid) gradients
    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation="sigmoid")

    # Hidden layers (relu) gradients
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent.
    
    Arguments:
    params -- dictionary containing parameters
    grads -- dictionary containing gradients
    learning_rate -- learning rate for gradient descent
    
    Returns:
    parameters -- dictionary with updated parameters
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters
