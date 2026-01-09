"""
Deep Learning Framework - Utility Functions
"""
import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """Create random mini-batches from (X, Y)."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

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
    return theta * x


def backward_propagation_scalar(x, theta):
    """Simple scalar backward propagation for gradient checking demo."""
    return x


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
            print("\033[93mBackward propagation error! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92mBackward propagation OK! difference = " + str(difference) + "\033[0m")

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
# SAVE / LOAD MODEL
# =============================================================================

def save_model(filepath, parameters, bn_params=None, bn_running=None):
    """
    Save model parameters to a .npz file.
    
    """
    save_dict = {'parameters': parameters}
    
    if bn_params is not None:
        save_dict['bn_params'] = bn_params
    if bn_running is not None:
        save_dict['bn_running'] = bn_running
    
    np.savez(filepath, **save_dict)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    """
    data = np.load(filepath, allow_pickle=True)
    
    parameters = data['parameters'].item()
    bn_params = data['bn_params'].item() if 'bn_params' in data else None
    bn_running = data['bn_running'].item() if 'bn_running' in data else None
    
    print(f"Model loaded from {filepath}")
    return parameters, bn_params, bn_running


# =============================================================================
# DATA SPLITTING
# =============================================================================

def train_val_test_split(X, Y, val_ratio=0.1, test_ratio=0.1, shuffle=True, seed=42):
    """
    Split data into training, validation, and test sets.
    """
    m = X.shape[1]
    
    if shuffle:
        np.random.seed(seed)
        permutation = np.random.permutation(m)
        X = X[:, permutation]
        Y = Y[:, permutation]
    
    # Calculate split indices
    val_size = int(m * val_ratio)
    test_size = int(m * test_ratio)
    train_size = m - val_size - test_size
    
    # Split
    X_train = X[:, :train_size]
    Y_train = Y[:, :train_size]
    
    X_val = X[:, train_size:train_size + val_size]
    Y_val = Y[:, train_size:train_size + val_size]
    
    X_test = X[:, train_size + val_size:]
    Y_test = Y[:, train_size + val_size:]
    
    print(f"Data split: train={train_size}, val={val_size}, test={test_size}")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def train_test_split(X, Y, test_ratio=0.2, shuffle=True, seed=42):
    """
    Split data into training and test sets only.
    
    """
    X_train, Y_train, _, _, X_test, Y_test = train_val_test_split(
        X, Y, val_ratio=0, test_ratio=test_ratio, shuffle=shuffle, seed=seed
    )
    return X_train, Y_train, X_test, Y_test
