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


# =============================================================================
# Helper Functions for Vectorized Convolution (im2col/col2im)
# =============================================================================

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Get index matrices for im2col transformation.
    Uses floor division to handle non-divisible dimensions (like PyTorch/TensorFlow).
    """
    N, C, H, W = x_shape
    
    # Use floor division - edge pixels that don't fit are dropped
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ 
    An implementation of im2col based on indexing.
    Note: Requires input to be (m, n_C, n_H, n_W) usually, but our framework uses (m, n_H, n_W, n_C).
    We will handle transposition inside the main functions.
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """
    An implementation of col2im based on indexing.
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
