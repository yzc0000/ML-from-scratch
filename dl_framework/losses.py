"""
Deep Learning Framework - Cost/Loss Functions den
"""
import numpy as np


def compute_cost(AL, Y, output, parameters=None, lambd=0):
    """
    Compute cost based on output type with optional L2 regularization.
    
    Arguments:
    AL -- Predictions from the model
    Y -- True labels/values
    output -- Type of output layer: "sigmoid", "softmax", or "linear"
    parameters -- Model parameters (for L2 regularization)
    lambd -- L2 regularization hyperparameter
    
    Returns:
    cost -- The computed cost
    """
    m = Y.shape[1]
    epsilon = 1e-15  # Small constant to prevent log(0)
    
    # Compute loss based on output type
    if output == "linear":
        # Mean Squared Error for regression
        mse_cost = 1./(2*m) * np.sum(np.square(AL - Y))
        base_cost = mse_cost
    elif output == "softmax":
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        cross_entropy = -1./m * np.sum(Y * np.log(AL_clipped))
        base_cost = cross_entropy
    else:  # sigmoid (binary)
        # Clip to prevent log(0)
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        logprobs = np.multiply(-np.log(AL_clipped), Y) + np.multiply(-np.log(1 - AL_clipped), 1 - Y)
        cross_entropy = 1./m * np.sum(logprobs)
        base_cost = cross_entropy
    
    # Add L2 regularization if specified
    if lambd > 0 and parameters is not None:
        L = len(parameters) // 2
        L2_sum = sum(np.sum(np.square(parameters['W' + str(l)])) for l in range(1, L + 1))
        L2_cost = (lambd / (2 * m)) * L2_sum
        return base_cost + L2_cost
    
    return base_cost
