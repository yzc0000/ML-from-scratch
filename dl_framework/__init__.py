"""
Deep Learning Framework

A modular deep learning library built from scratch with NumPy.

Usage:
    from dl_framework import model, predict, accuracy
    
    # Binary classification
    params, _, costs = model(X, Y, [784, 64, 1], output="sigmoid", optimizer="adam")
    
    # Multi-class classification  
    params, _, costs = model(X, Y, [784, 64, 10], output="softmax", optimizer="adam")
    
    # With regularization
    params, _, costs = model(X, Y, [784, 64, 1], lambd=0.1)  # L2
    params, _, costs = model(X, Y, [784, 64, 1], keep_prob=0.8)  # Dropout
    
    # With batch normalization
    params, bn, costs = model(X, Y, [784, 64, 1], use_batchnorm=True)
"""

# Activations
from .activations import (
    sigmoid, relu, softmax, tanh_activation, linear_activation,
    sigmoid_backward, relu_backward, softmax_backward, tanh_backward, linear_backward
)

# Initialization
from .initialization import initialize_parameters, initialize_bn_parameters

# Layers
from .layers import L_model_forward, L_model_backward, linear_forward, linear_backward

# Losses
from .losses import compute_cost

# Optimizers
from .optimizers import (
    update_parameters_with_gd,
    initialize_velocity, update_parameters_with_momentum,
    initialize_adam, update_parameters_with_adam
)

# Regularization
from .regularization import (
    batchnorm_forward, batchnorm_backward,
    update_bn_parameters,
    initialize_bn_running_stats,
    dropout_forward, dropout_backward
)

# Utils
from .utils import (
    random_mini_batches,
    update_lr, schedule_lr_decay,
    gradient_check,
    dictionary_to_vector, vector_to_dictionary, gradients_to_vector,
    save_model, load_model,
    train_val_test_split, train_test_split
)

# Models
from .models import model, predict, accuracy, cnn_model, cnn_predict

# CNN Layers (included in layers.py)
from .layers import (
    conv_forward, conv_backward,
    pool_forward, pool_backward,
    flatten_forward, flatten_backward,
    residual_forward, residual_backward,
    CNN_model_forward, CNN_model_backward
)

# CNN Initialization and Optimizers
from .initialization import (
    initialize_conv_parameters, initialize_cnn_parameters,
    initialize_cnn_bn_parameters, initialize_cnn_bn_running_stats
)


