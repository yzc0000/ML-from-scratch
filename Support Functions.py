import numpy as np
import math
import matplotlib.pyplot as plt


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(Z):
    """Compute sigmoid activation."""
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """Compute ReLU activation."""
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """Backward propagation for sigmoid."""
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """Backward propagation for ReLU."""
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax(Z):
    """
    Compute softmax activation for multi-class classification.
    
    Arguments:
    Z -- pre-activation values, shape (num_classes, m)
    
    Returns:
    A -- softmax output (probabilities), shape (num_classes, m)
    cache -- Z for backward pass
    """
    # Subtract max for numerical stability
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache = Z
    return A, cache


def softmax_backward(dA, cache):
    """
    Backward propagation for softmax.
    Note: When used with categorical cross-entropy, dZ = A - Y directly.
    This function is for standalone softmax gradient if needed.
    """
    Z = cache
    A, _ = softmax(Z)
    # For softmax with cross-entropy, typically use dZ = A - Y directly
    # This is the general Jacobian-based gradient
    m = Z.shape[1]
    dZ = A * (dA - np.sum(dA * A, axis=0, keepdims=True))
    return dZ


def tanh_activation(Z):
    """Compute tanh activation."""
    A = np.tanh(Z)
    cache = Z
    return A, cache


def tanh_backward(dA, cache):
    """Backward propagation for tanh."""
    Z = cache
    A = np.tanh(Z)
    dZ = dA * (1 - A**2)
    return dZ


# =============================================================================
# PARAMETER INITIALIZATION
# =============================================================================

def initialize_parameters(layers_dims):
    """
    Initialize parameters for an L-layer neural network using He initialization.
    
    Arguments:
    layers_dims -- list containing the dimensions of each layer
    
    Returns:
    parameters -- dictionary containing W1, b1, ..., WL, bL
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_zeros(layers_dims):
    """Initialize all parameters to zeros (not recommended, for demonstration only)."""
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    """Initialize parameters with random values scaled by 10."""
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    """Initialize parameters using He initialization."""
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1

    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


# =============================================================================
# BATCH NORMALIZATION
# =============================================================================

def initialize_bn_parameters(layers_dims):
    """
    Initialize batch normalization parameters (gamma and beta) for each layer.
    
    Arguments:
    layers_dims -- list containing dimensions of each layer
    
    Returns:
    bn_params -- dictionary containing gamma and beta for each layer
    """
    bn_params = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        bn_params['gamma' + str(l)] = np.ones((layers_dims[l], 1))
        bn_params['beta' + str(l)] = np.zeros((layers_dims[l], 1))
    
    return bn_params


def batchnorm_forward(Z, gamma, beta, epsilon=1e-8):
    """
    Batch normalization forward pass.
    
    Arguments:
    Z -- pre-activation values, shape (n, m)
    gamma -- scale parameter, shape (n, 1)
    beta -- shift parameter, shape (n, 1)
    epsilon -- small constant for numerical stability
    
    Returns:
    Z_norm -- normalized and scaled Z
    cache -- values needed for backward pass
    """
    m = Z.shape[1]
    
    mu = np.mean(Z, axis=1, keepdims=True)
    Z_centered = Z - mu
    var = np.var(Z, axis=1, keepdims=True)
    Z_hat = Z_centered / np.sqrt(var + epsilon)
    Z_norm = gamma * Z_hat + beta
    
    cache = (Z, Z_hat, Z_centered, mu, var, gamma, beta, epsilon)
    return Z_norm, cache


def batchnorm_backward(dZ_norm, cache):
    """
    Batch normalization backward pass.
    
    Arguments:
    dZ_norm -- gradient of loss w.r.t. normalized output
    cache -- values from forward pass
    
    Returns:
    dZ -- gradient w.r.t. input Z
    dgamma -- gradient w.r.t. gamma
    dbeta -- gradient w.r.t. beta
    """
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


# =============================================================================
# L-LAYER FORWARD PROPAGATION
# =============================================================================

def linear_forward(A, W, b):
    """Compute Z = W @ A + b."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for LINEAR->ACTIVATION layer.
    
    Arguments:
    A_prev -- activations from previous layer
    W -- weights
    b -- bias
    activation -- "sigmoid" or "relu"
    
    Returns:
    A -- post-activation value
    cache -- tuple of (linear_cache, activation_cache)
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    L-layer forward propagation: [LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID
    
    Arguments:
    X -- input data, shape (input_size, m)
    parameters -- dictionary containing W1, b1, ..., WL, bL
    
    Returns:
    AL -- output of the last layer
    caches -- list of caches for each layer
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    # Hidden layers with ReLU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], 
                                              parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    # Output layer with Sigmoid
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                           parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    return AL, caches


def L_model_forward_with_dropout(X, parameters, keep_prob=0.5):
    """
    L-layer forward propagation with dropout.
    [LINEAR->RELU+DROPOUT]*(L-1) -> LINEAR->SIGMOID
    
    Arguments:
    X -- input data
    parameters -- W and b for each layer
    keep_prob -- probability of keeping a neuron
    
    Returns:
    AL -- output of the last layer
    caches -- list of caches for each layer (includes dropout masks)
    """
    np.random.seed(1)
    caches = []
    A = X
    L = len(parameters) // 2
    
    # Hidden layers with ReLU + Dropout
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
        # Apply dropout
        D = np.random.rand(*A.shape)
        D = (D < keep_prob).astype(int)
        A = A * D / keep_prob
        
        cache = (linear_cache, activation_cache, D)
        caches.append(cache)
    
    # Output layer (no dropout)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                           parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    return AL, caches


def L_model_forward_with_batchnorm(X, parameters, bn_params, training=True):
    """
    L-layer forward propagation with batch normalization.
    [LINEAR->BATCHNORM->RELU]*(L-1) -> LINEAR->SIGMOID
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    # Hidden layers: LINEAR -> BATCHNORM -> RELU
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        gamma = bn_params['gamma' + str(l)]
        beta = bn_params['beta' + str(l)]
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        Z_norm, bn_cache = batchnorm_forward(Z, gamma, beta)
        A, activation_cache = relu(Z_norm)
        
        cache = (linear_cache, bn_cache, activation_cache)
        caches.append(cache)
    
    # Output layer: LINEAR -> SIGMOID (no batch norm)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], 
                                           parameters['b' + str(L)], "sigmoid")
    caches.append(cache)
    
    return AL, caches


# =============================================================================
# L-LAYER BACKWARD PROPAGATION
# =============================================================================

def linear_backward(dZ, cache):
    """Backward propagation for linear layer."""
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Backward propagation for LINEAR->ACTIVATION layer.
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    L-layer backward propagation.
    
    Arguments:
    AL -- output of forward propagation
    Y -- true labels
    caches -- list of caches from forward propagation
    
    Returns:
    grads -- dictionary with gradients
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Derivative of cost w.r.t. AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Output layer (sigmoid)
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Hidden layers (relu)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, "relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads


def L_model_backward_with_dropout(AL, Y, caches, keep_prob):
    """
    L-layer backward propagation with dropout.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Derivative of cost w.r.t. AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Output layer (no dropout)
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Hidden layers with dropout
    for l in reversed(range(L - 1)):
        linear_cache, activation_cache, D = caches[l]
        
        # Apply dropout mask
        dA = grads["dA" + str(l + 1)]
        dA = dA * D / keep_prob
        
        # ReLU backward
        dZ = relu_backward(dA, activation_cache)
        
        # Linear backward
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    
    return grads


def L_model_backward_with_regularization(AL, Y, caches, parameters, lambd):
    """
    L-layer backward propagation with L2 regularization.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Derivative of cost w.r.t. AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Output layer (sigmoid)
    current_cache = caches[L - 1]
    linear_cache, activation_cache = current_cache
    dZ = sigmoid_backward(dAL, activation_cache)
    A_prev, W, b = linear_cache
    
    grads["dW" + str(L)] = 1/m * np.dot(dZ, A_prev.T) + (lambd / m) * parameters['W' + str(L)]
    grads["db" + str(L)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dA" + str(L-1)] = np.dot(W.T, dZ)
    
    # Hidden layers (relu)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        linear_cache, activation_cache = current_cache
        
        dZ = relu_backward(grads["dA" + str(l + 1)], activation_cache)
        A_prev, W, b = linear_cache
        
        grads["dW" + str(l + 1)] = 1/m * np.dot(dZ, A_prev.T) + (lambd / m) * parameters['W' + str(l + 1)]
        grads["db" + str(l + 1)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l)] = np.dot(W.T, dZ)
    
    return grads


def L_model_backward_with_batchnorm(AL, Y, caches, bn_params):
    """
    L-layer backward propagation with batch normalization.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    # Derivative of cost w.r.t. AL
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Output layer (no batch norm)
    current_cache = caches[L - 1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Hidden layers with batch norm
    for l in reversed(range(L - 1)):
        linear_cache, bn_cache, activation_cache = caches[l]
        
        # ReLU backward
        dZ_norm = relu_backward(grads["dA" + str(l + 1)], activation_cache)
        
        # Batch norm backward
        dZ, dgamma, dbeta = batchnorm_backward(dZ_norm, bn_cache)
        grads["dgamma" + str(l + 1)] = dgamma
        grads["dbeta" + str(l + 1)] = dbeta
        
        # Linear backward
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    
    return grads


# =============================================================================
# COST FUNCTIONS
# =============================================================================

def compute_cost(AL, Y):
    """Compute cross-entropy cost."""
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    """Compute cost with L2 regularization (L-layer compatible)."""
    m = Y.shape[1]
    L = len(parameters) // 2
    
    cross_entropy_cost = compute_cost(AL, Y)
    
    L2_sum = 0
    for l in range(1, L + 1):
        L2_sum += np.sum(np.square(parameters['W' + str(l)]))
    
    L2_regularization_cost = (lambd / (2 * m)) * L2_sum
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


def compute_cost_categorical(AL, Y):
    """
    Compute categorical cross-entropy cost for multi-class classification.
    
    Arguments:
    AL -- softmax output, shape (num_classes, m)
    Y -- one-hot encoded labels, shape (num_classes, m)
    
    Returns:
    cost -- categorical cross-entropy cost
    """
    m = Y.shape[1]
    # Clip to prevent log(0)
    AL_clipped = np.clip(AL, 1e-15, 1 - 1e-15)
    cost = -1./m * np.sum(Y * np.log(AL_clipped))
    return cost


def compute_cost_categorical_with_regularization(AL, Y, parameters, lambd):
    """Compute categorical cross-entropy cost with L2 regularization."""
    m = Y.shape[1]
    L = len(parameters) // 2
    
    cross_entropy_cost = compute_cost_categorical(AL, Y)
    
    L2_sum = 0
    for l in range(1, L + 1):
        L2_sum += np.sum(np.square(parameters['W' + str(l)]))
    
    L2_regularization_cost = (lambd / (2 * m)) * L2_sum
    cost = cross_entropy_cost + L2_regularization_cost
    return cost


# =============================================================================
# PARAMETER UPDATE METHODS
# =============================================================================

def update_parameters_with_gd(parameters, grads, learning_rate):
    """Update parameters using gradient descent."""
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters


def update_bn_parameters(bn_params, grads, learning_rate):
    """Update batch normalization parameters."""
    L = len(bn_params) // 2

    for l in range(1, L + 1):
        if 'dgamma' + str(l) in grads:
            bn_params['gamma' + str(l)] -= learning_rate * grads['dgamma' + str(l)]
            bn_params['beta' + str(l)] -= learning_rate * grads['dbeta' + str(l)]

    return bn_params


# =============================================================================
# MOMENTUM OPTIMIZER
# =============================================================================

def initialize_velocity(parameters):
    """Initialize velocity for momentum optimizer."""
    L = len(parameters) // 2
    v = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

    return v


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """Update parameters using momentum."""
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * v["db" + str(l)]

    return parameters, v


# =============================================================================
# ADAM OPTIMIZER
# =============================================================================

def initialize_adam(parameters):
    """Initialize v and s for Adam optimizer."""
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Update parameters using Adam optimizer."""
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - np.power(beta1, t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - np.power(beta1, t))

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - np.power(beta2, t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - np.power(beta2, t))

        parameters["W" + str(l)] -= learning_rate * v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)]) + epsilon)
        parameters["b" + str(l)] -= learning_rate * v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)]) + epsilon)

    return parameters, v, s, v_corrected, s_corrected


# =============================================================================
# MINI-BATCHES
# =============================================================================

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """Create random mini-batches from (X, Y)."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(num_complete_minibatches):
        i = k * mini_batch_size
        j = (k + 1) * mini_batch_size
        mini_batch_X = shuffled_X[:, i:j]
        mini_batch_Y = shuffled_Y[:, i:j]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        i = num_complete_minibatches * mini_batch_size
        mini_batch_X = shuffled_X[:, i:m]
        mini_batch_Y = shuffled_Y[:, i:m]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


# =============================================================================
# LEARNING RATE DECAY
# =============================================================================

def update_lr(learning_rate0, epoch_num, decay_rate):
    """Update learning rate using inverse time decay."""
    learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)
    return learning_rate


def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """Update learning rate with scheduled decay."""
    learning_rate = learning_rate0 / (1 + decay_rate * (epoch_num // time_interval))
    return learning_rate


# =============================================================================
# GRADIENT CHECKING UTILITIES
# =============================================================================

def forward_propagation_scalar(x, theta):
    """Simple scalar forward propagation for gradient checking demo."""
    J = theta * x
    return J


def backward_propagation_scalar(x, theta):
    """Simple scalar backward propagation for gradient checking demo."""
    dtheta = x
    return dtheta


def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    """Gradient checking for a simple scalar function."""
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation_scalar(x, theta_plus)
    J_minus = forward_propagation_scalar(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation_scalar(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93mThere is a mistake in backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92mYour backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


def dictionary_to_vector(parameters):
    """Roll all parameters into a single vector."""
    keys = []
    count = 0
    L = len(parameters) // 2

    for l in range(1, L + 1):
        for param_type in ["W", "b"]:
            key = param_type + str(l)
            new_vector = np.reshape(parameters[key], (-1, 1))
            keys.extend([(key, i) for i in range(new_vector.shape[0])])

            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count += 1

    return theta, keys


def vector_to_dictionary(theta, layers_dims):
    """Unroll a vector back to parameters dictionary."""
    parameters = {}
    L = len(layers_dims)
    idx = 0

    for l in range(1, L):
        w_shape = (layers_dims[l], layers_dims[l-1])
        w_size = layers_dims[l] * layers_dims[l-1]
        parameters["W" + str(l)] = theta[idx:idx + w_size].reshape(w_shape)
        idx += w_size

        b_shape = (layers_dims[l], 1)
        b_size = layers_dims[l]
        parameters["b" + str(l)] = theta[idx:idx + b_size].reshape(b_shape)
        idx += b_size

    return parameters


def gradients_to_vector(gradients):
    """Roll all gradients into a single vector."""
    count = 0
    keys = sorted([k for k in gradients.keys() if k.startswith('dW') or k.startswith('db')])

    for key in keys:
        new_vector = np.reshape(gradients[key], (-1, 1))
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count += 1

    return theta


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def model(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000,
          print_cost=True, decay=None, decay_rate=1):
    """
    L-layer neural network with various optimizer options.

    Arguments:
    X -- input data of shape (input size, number of examples)
    Y -- true labels of shape (1, number of examples)
    layers_dims -- list containing the size of each layer
    optimizer -- "gd", "momentum", or "adam"
    learning_rate -- learning rate
    mini_batch_size -- size of mini batches
    beta -- momentum hyperparameter
    beta1 -- Adam hyperparameter for first moment
    beta2 -- Adam hyperparameter for second moment
    epsilon -- Adam hyperparameter to prevent division by zero
    num_epochs -- number of training epochs
    print_cost -- True to print cost every 1000 epochs
    decay -- decay function (e.g., update_lr or schedule_lr_decay)
    decay_rate -- decay rate for learning rate scheduling

    Returns:
    parameters -- learned parameters
    """
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]
    learning_rate0 = learning_rate

    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            AL, caches = L_model_forward(minibatch_X, parameters)
            cost_total += compute_cost(AL, minibatch_Y)
            grads = L_model_backward(AL, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s, _, _ = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )

        cost_avg = cost_total / m

        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
            if decay:
                print("Learning rate after epoch %i: %f" % (i, learning_rate))

        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def model_with_regularization(X, Y, layers_dims, learning_rate=0.3, num_iterations=30000,
                               print_cost=True, lambd=0, keep_prob=1):
    """
    L-layer neural network with L2 regularization and/or dropout.

    Arguments:
    X -- input data
    Y -- true labels
    layers_dims -- list containing size of each layer
    learning_rate -- learning rate
    num_iterations -- number of iterations
    print_cost -- True to print cost every 10000 iterations
    lambd -- L2 regularization hyperparameter (0 = no regularization)
    keep_prob -- dropout keep probability (1 = no dropout)

    Returns:
    parameters -- learned parameters
    """
    costs = []
    m = X.shape[1]

    parameters = initialize_parameters(layers_dims)

    for i in range(num_iterations):
        # Forward propagation
        if keep_prob == 1:
            AL, caches = L_model_forward(X, parameters)
        else:
            AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob)

        # Compute cost
        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

        # Backward propagation
        assert (lambd == 0 or keep_prob == 1), "Can only use one of L2 or dropout at a time"

        if lambd == 0 and keep_prob == 1:
            grads = L_model_backward(AL, Y, caches)
        elif lambd != 0:
            grads = L_model_backward_with_regularization(AL, Y, caches, parameters, lambd)
        else:
            grads = L_model_backward_with_dropout(AL, Y, caches, keep_prob)

        # Update parameters
        parameters = update_parameters_with_gd(parameters, grads, learning_rate)

        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def model_with_batchnorm(X, Y, layers_dims, optimizer="adam", learning_rate=0.001,
                          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
                          epsilon=1e-8, num_epochs=5000, print_cost=True):
    """
    L-layer neural network with batch normalization.
    """
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]

    parameters = initialize_parameters(layers_dims)
    bn_params = initialize_bn_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            AL, caches = L_model_forward_with_batchnorm(minibatch_X, parameters, bn_params)
            cost_total += compute_cost(AL, minibatch_Y)
            grads = L_model_backward_with_batchnorm(AL, minibatch_Y, caches, bn_params)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
                bn_params = update_bn_parameters(bn_params, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                bn_params = update_bn_parameters(bn_params, grads, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s, _, _ = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )
                bn_params = update_bn_parameters(bn_params, grads, learning_rate)

        cost_avg = cost_total / m

        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))

        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters, bn_params
