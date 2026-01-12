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
          lambd=0, keep_prob=1.0, use_batchnorm=False, weight_decay=0,
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
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon, weight_decay
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
    elif output == "linear":
        # For regression, return raw predicted values directly
        predictions = AL
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
from .initialization import initialize_cnn_parameters, initialize_cnn_bn_parameters, initialize_cnn_bn_running_stats



def cnn_model(X, Y, layers,
              optimizer="adam", learning_rate=0.001,
              beta1=0.9, beta2=0.999, epsilon=1e-8,
              mini_batch_size=64, num_epochs=10,
              lr_decay=None, decay_rate=0,
              lambd=0, keep_prob=1.0, weight_decay=0,  # Regularization
              use_batchnorm=False,  # Batch Normalization
              print_cost=True, print_interval=1, print_lr=True, plot_cost=True):
    """
    Train a Convolutional Neural Network.
    
    Arguments:
    X -- Input data, shape (m, n_H, n_W, n_C)
    Y -- Labels, shape (num_classes, m)
    layers -- List of layer configurations
    optimizer -- Currently only "adam" supported
    learning_rate -- Learning rate for optimizer
    beta1, beta2, epsilon -- Adam hyperparameters
    mini_batch_size -- Size of mini-batches
    num_epochs -- Number of training epochs
    lr_decay -- Learning rate decay function (optional)
    decay_rate -- Decay rate for lr_decay
    lambd -- L2 regularization hyperparameter (adds to gradients)
    keep_prob -- Dropout keep probability (1.0 = no dropout)
    weight_decay -- Decoupled weight decay (AdamW style, applied separately)
    use_batchnorm -- Whether to use batch normalization after conv layers
    print_cost -- Whether to print cost during training
    print_interval -- How often to print cost
    print_lr -- Whether to print learning rate
    plot_cost -- Whether to plot cost curve
    
    Returns:
    parameters -- Trained model parameters
    costs -- Training cost history
    bn_params -- BN parameters (if use_batchnorm=True, else None)
    bn_running -- BN running stats (if use_batchnorm=True, else None)
    
    Note: Use EITHER lambd OR weight_decay, not both. weight_decay is preferred with Adam.
    """
    m = X.shape[0]
    input_shape = X.shape[1:]  # (n_H, n_W, n_C)
    costs = []
    t = 0
    learning_rate0 = learning_rate
    
    # Initialize parameters
    parameters = initialize_cnn_parameters(layers, input_shape)
    v, s = initialize_adam(parameters)
    
    # Initialize batch normalization (optional)
    if use_batchnorm:
        bn_params = initialize_cnn_bn_parameters(layers, input_shape)
        bn_running = initialize_cnn_bn_running_stats(layers, input_shape)
    else:
        bn_params = None
        bn_running = None
    
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
            
            # Forward propagation (training=True for dropout and BN)
            AL, caches = CNN_model_forward(
                X_batch, layers, parameters, 
                keep_prob=keep_prob, training=True,
                bn_params=bn_params, bn_running=bn_running
            )
            
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
            grads = CNN_model_backward(
                AL, Y_batch, layers, caches, 
                keep_prob=keep_prob, use_batchnorm=use_batchnorm
            )
            
            # Add L2 Gradient Regularization
            if lambd > 0:
                for key in grads.keys():
                    if key.startswith('dW'):
                        # dW_conv1 -> W_conv1, dW_dense1 -> W_dense1
                        param_name = key[1:]  # Remove leading 'd'
                        if param_name in parameters:
                             grads[key] += (lambd / (end - start)) * parameters[param_name]
            
            # Update parameters (Adam with optional weight decay)
            t += 1
            parameters, v, s = update_parameters_with_adam(
                parameters, grads, v, s, t,
                learning_rate, beta1, beta2, epsilon, weight_decay
            )
            
            # Update batch norm parameters (gamma, beta)
            if use_batchnorm:
                for key in bn_params.keys():
                    grad_key = 'd' + key
                    if grad_key in grads:
                        bn_params[key] -= learning_rate * grads[grad_key]
        
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
    
    return parameters, costs, bn_params, bn_running


def cnn_predict(X, layers, parameters, bn_params=None, bn_running=None):
    """
    Make predictions using trained CNN parameters.
    
    Arguments:
    X -- Input data, shape (m, n_H, n_W, n_C)
    layers -- Layer configurations (same as used in training)
    parameters -- Trained parameters from cnn_model()
    bn_params -- Batch norm parameters (optional, from training)
    bn_running -- Batch norm running stats (optional, from training)
    
    Returns:
    predictions -- Predicted class indices
    probabilities -- Class probabilities
    """
    AL, _ = CNN_model_forward(
        X, layers, parameters,
        keep_prob=1.0, training=False,  # Inference mode
        bn_params=bn_params, bn_running=bn_running
    )
    predictions = np.argmax(AL, axis=0)
    return predictions, AL
