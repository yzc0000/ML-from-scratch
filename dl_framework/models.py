"""
Deep Learning Framework - Model Training Functions
"""
import numpy as np
import matplotlib.pyplot as plt

from .initialization import initialize_parameters, initialize_bn_parameters
from .layers import L_model_forward, L_model_backward
from .losses import compute_cost
from .optimizers import (
    update_parameters_with_gd, 
    initialize_velocity, update_parameters_with_momentum,
    initialize_adam, update_parameters_with_adam
)
from .regularization import update_bn_parameters, initialize_bn_running_stats
from .utils import random_mini_batches


def model(X, Y, layers_dims, 
          output="sigmoid",
          lambd=0, keep_prob=1.0, use_batchnorm=False,
          optimizer="adam", learning_rate=0.001,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          mini_batch_size=64, num_epochs=5000,
          lr_decay=None, decay_rate=0,
          init_method="he", print_cost=True, plot_cost=True):
          
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]
    learning_rate0 = learning_rate

    # Initialize parameters
    parameters = initialize_parameters(layers_dims, method=init_method)
    
    # Initialize batch norm
    if use_batchnorm:
        bn_params = initialize_bn_parameters(layers_dims)
        bn_running = initialize_bn_running_stats(layers_dims)
    else:
        bn_params = None
        bn_running = None

    # Initialize optimizer state
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Training loop
    for epoch in range(num_epochs):
        seed += 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation (training=True)
            AL, caches = L_model_forward(
                minibatch_X, parameters,
                output=output,
                keep_prob=keep_prob,
                bn_params=bn_params,
                bn_running=bn_running,
                training=True
            )
            
            # Compute cost
            cost_total += compute_cost(AL, minibatch_Y, output=output, 
                                       parameters=parameters, lambd=lambd)
            
            # Backward propagation
            grads = L_model_backward(
                AL, minibatch_Y, caches, parameters,
                output=output,
                keep_prob=keep_prob,
                lambd=lambd,
                bn_params=bn_params
            )

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                parameters, v, s, _, _ = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )
            
            # Update batch norm parameters
            if use_batchnorm:
                bn_params = update_bn_parameters(bn_params, grads, learning_rate)

        cost_avg = cost_total / m

        # Learning rate decay
        if lr_decay is not None:
            learning_rate = lr_decay(learning_rate0, epoch, decay_rate)

        # Logging
        if print_cost and epoch % 1000 == 0:
            print(f"Cost after epoch {epoch}: {cost_avg:.6f}")
            if lr_decay:
                print(f"  Learning rate: {learning_rate:.6f}")

        if epoch % 100 == 0:
            costs.append(cost_avg)

    # Plot cost
    if plot_cost:
        plt.plot(costs)
        plt.ylabel('Cost')
        plt.xlabel('Epochs (per 100)')
        plt.title(f"Learning rate = {learning_rate}")
        plt.show()

    return parameters, bn_params, bn_running, costs


def predict(X, parameters, output="sigmoid", bn_params=None, bn_running=None):
    """
    Make predictions using trained parameters (inference mode).
    """
    # Forward pass with training=False (uses running stats for batch norm)
    AL, _ = L_model_forward(
        X, parameters, 
        output=output,
        keep_prob=1.0,  # No dropout during inference
        bn_params=bn_params,
        bn_running=bn_running,
        training=False  # Use running mean/var for batch norm
    )
    
    if output == "softmax":
        predictions = np.argmax(AL, axis=0)
    else:  # sigmoid
        predictions = (AL > 0.5).astype(int)
    
    return predictions, AL


def accuracy(predictions, Y):
    """Calculate accuracy."""
    if Y.shape[0] > 1:  # One-hot encoded (multiclass)
        Y_labels = np.argmax(Y, axis=0)
        return np.mean(predictions == Y_labels) * 100
    else:  # Binary
        return np.mean(predictions == Y) * 100
