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
          init_method="he", print_cost=True, print_interval=100, print_lr=True, plot_cost=True):
          
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
                parameters, v, s = update_parameters_with_adam(
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
        if print_cost and epoch % print_interval == 0:
            print(f"Cost after epoch {epoch}: {cost_avg:.6f}")
            if lr_decay and print_lr:
                print(f"  Learning rate: {learning_rate:.6f}")

        if epoch % print_interval == 0:
            costs.append(cost_avg)

    # Plot cost
    if plot_cost:
        plt.plot(costs)
        plt.ylabel('Cost')
        plt.xlabel(f'Epochs (per {print_interval})')
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


# =============================================================================
# CNN Model Function
# =============================================================================

from .layers import CNN_model_forward, CNN_model_backward
from .initialization import initialize_cnn_parameters



def cnn_model(X, Y, layers,
              optimizer="adam", learning_rate=0.001,
              beta1=0.9, beta2=0.999, epsilon=1e-8,
              mini_batch_size=64, num_epochs=10,
              lr_decay=None, decay_rate=0,
              lambd=0, keep_prob=1.0,  # Regularization
              print_cost=True, print_interval=1, print_lr=True, plot_cost=True):
    """
    Train a Convolutional Neural Network.
    ...
    lambd -- L2 regularization hyperparameter
    keep_prob -- Dropout keep probability (1.0 = no dropout)
    ...
    """
    m = X.shape[0]
    input_shape = X.shape[1:]  # (n_H, n_W, n_C)
    costs = []
    t = 0
    learning_rate0 = learning_rate
    
    # Initialize parameters
    parameters = initialize_cnn_parameters(layers, input_shape)
    v, s = initialize_adam(parameters)
    
    # Training loop
    for epoch in range(num_epochs):
        # Shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[:, permutation]
        
        cost_total = 0
        num_batches = int(np.ceil(m / mini_batch_size))
        
        for batch_idx in range(num_batches):
            start = batch_idx * mini_batch_size
            end = min(start + mini_batch_size, m)
            
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[:, start:end]
            
            # Forward propagation (training=True for dropout)
            AL, caches = CNN_model_forward(X_batch, layers, parameters, keep_prob=keep_prob, training=True)
            
            # Compute cost (cross-entropy + L2)
            eps = 1e-15
            AL_clipped = np.clip(AL, eps, 1 - eps)
            cross_entropy_cost = -np.sum(Y_batch * np.log(AL_clipped)) / (end - start)
            
            # L2 Cost
            L2_regularization_cost = 0
            if lambd > 0:
                for key in parameters.keys():
                    if key.startswith('W'):
                        L2_regularization_cost += np.sum(np.square(parameters[key]))
                L2_regularization_cost *= (lambd / (2 * (end - start)))
            
            cost = cross_entropy_cost + L2_regularization_cost
            cost_total += cost
            
            # Backward propagation
            grads = CNN_model_backward(AL, Y_batch, layers, caches, keep_prob=keep_prob)
            
            # Add L2 Gradient Regularization
            if lambd > 0:
                for key in grads.keys():
                    if key.startswith('dW'):
                        # dW_conv1 -> W_conv1, dW_dense1 -> W_dense1
                        param_name = key[1:]  # Remove leading 'd'
                        if param_name in parameters:
                             grads[key] += (lambd / (end - start)) * parameters[param_name]
            
            # Update parameters (Adam)
            t += 1
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, t,
                learning_rate, beta1, beta2, epsilon
            )
        
        cost_avg = cost_total / num_batches
        
        # Learning rate decay
        if lr_decay is not None:
            learning_rate = lr_decay(learning_rate0, epoch, decay_rate)
        
        # Logging
        if print_cost and epoch % print_interval == 0:
            print(f"Epoch {epoch}: cost = {cost_avg:.6f}")
            if lr_decay and print_lr:
                print(f"  Learning rate: {learning_rate:.6f}")
            costs.append(cost_avg)
    
    # Plot cost
    if plot_cost:
        plt.plot(costs)
        plt.ylabel('Cost')
        plt.xlabel(f'Epochs (per {print_interval})')
        plt.title(f"CNN Training - Learning rate = {learning_rate}")
        plt.show()
    
    return parameters, costs


def cnn_predict(X, layers, parameters):
    """
    Make predictions using trained CNN parameters.
    
    Arguments:
    X -- Input data, shape (m, n_H, n_W, n_C)
    layers -- Layer configurations (same as used in training)
    parameters -- Trained parameters from cnn_model()
    
    Returns:
    predictions -- Predicted class indices
    probabilities -- Class probabilities
    """
    AL, _ = CNN_model_forward(X, layers, parameters)
    predictions = np.argmax(AL, axis=0)
    return predictions, AL
