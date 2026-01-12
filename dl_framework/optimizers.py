"""
Deep Learning Framework - Optimizers
"""
import numpy as np


def update_parameters_with_gd(parameters, grads, learning_rate):
    """Update parameters using gradient descent."""
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters


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


def initialize_adam(parameters):
    """
    Initialize v and s for Adam optimizer.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    
    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
    """
    v = {}
    s = {}

    for key in parameters:
        v[key] = np.zeros_like(parameters[key])
        s[key] = np.zeros_like(parameters[key])

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
    """
    Update parameters using Adam optimizer with optional weight decay (AdamW).
    
    Arguments:
    parameters -- dict of parameters (W1, b1, ..., or W_conv1, etc.)
    grads -- dict of gradients (keys prefixed with 'd')
    v -- dict of velocity (first moment)
    s -- dict of squared velocity (second moment)
    t -- Adam timestep
    learning_rate -- step size
    beta1 -- exponential decay for first moment
    beta2 -- exponential decay for second moment
    epsilon -- small value for numerical stability
    weight_decay -- decoupled weight decay coefficient (AdamW style, default 0)
    
    Returns:
    parameters -- updated parameters
    v -- updated velocity
    s -- updated squared velocity
    
    Note:
    - weight_decay applies decoupled weight decay (AdamW): W = W - lr*weight_decay*W
    - This is different from L2 regularization (lambd) which adds to gradients
    - For best results, use EITHER lambd (L2 reg) OR weight_decay, not both
    """
    for key in parameters:
        grad_key = "d" + key
        # Only update if gradient exists (some params might be fixed/unused)
        if grad_key in grads:
            # Update biased first moment
            v[key] = beta1 * v[key] + (1 - beta1) * grads[grad_key]
            
            # Update biased second moment
            s[key] = beta2 * s[key] + (1 - beta2) * (grads[grad_key] ** 2)

            # Bias correction
            v_corrected = v[key] / (1 - beta1 ** t)
            s_corrected = s[key] / (1 - beta2 ** t)

            # Update parameters (Adam step)
            parameters[key] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
            
            # Apply decoupled weight decay (AdamW) - only to weights, not biases
            if weight_decay > 0 and key.startswith('W'):
                parameters[key] -= learning_rate * weight_decay * parameters[key]

    return parameters, v, s

