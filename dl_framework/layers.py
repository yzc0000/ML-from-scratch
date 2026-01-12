"""
Deep Learning Framework - Forward/Backward Propagation Layers
"""
import numpy as np

from .activations import sigmoid, relu, softmax, sigmoid_backward, relu_backward, linear_activation, linear_backward
from .regularization import batchnorm_forward, batchnorm_backward, dropout_forward, dropout_backward


def linear_forward(A, W, b):
    """Compute Z = W @ A + b."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_backward(dZ, cache, lambd=0, W=None):
    """Backward propagation for linear layer with optional L2 regularization."""
    A_prev, W_cache, b = cache
    m = A_prev.shape[1]
    
    dW = 1/m * np.dot(dZ, A_prev.T)
    if lambd > 0:
        dW += (lambd / m) * W_cache
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W_cache.T, dZ)
    
    return dA_prev, dW, db


def L_model_forward(X, parameters, output="sigmoid", keep_prob=1.0, 
                    bn_params=None, bn_running=None, training=True):
    """
    L-layer forward propagation with configurable options.
    """
    caches = []
    A = X
    L = len(parameters) // 2
    use_dropout = keep_prob < 1.0 and training  # Only dropout during training
    use_batchnorm = bn_params is not None
    
    if use_dropout:
        np.random.seed(1)
    
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        # Linear
        Z, linear_cache = linear_forward(A_prev, W, b)
        
        # Batch Normalization (optional)
        if use_batchnorm:
            gamma = bn_params['gamma' + str(l)]
            beta = bn_params['beta' + str(l)]
            Z, bn_cache = batchnorm_forward(
                Z, gamma, beta, 
                bn_running=bn_running, 
                layer_idx=l, 
                training=training
            )
        else:
            bn_cache = None
        
        A, activation_cache = relu(Z)
        
        if use_dropout:
            A, D_cache = dropout_forward(A, keep_prob)
            # D_cache is (D, keep_prob) tuple
        else:
            D_cache = None
        
        cache = (linear_cache, activation_cache, bn_cache, D_cache)
        caches.append(cache)
    
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    Z, linear_cache = linear_forward(A, W, b)
    
    if output == "softmax":
        AL, activation_cache = softmax(Z)
    elif output == "linear":
        AL, activation_cache = linear_activation(Z)
    else:  # sigmoid
        AL, activation_cache = sigmoid(Z)
    
    cache = (linear_cache, activation_cache, None, None)
    caches.append(cache)
    
    return AL, caches


def L_model_backward(AL, Y, caches, parameters, output="sigmoid", keep_prob=1.0, 
                     lambd=0, bn_params=None):
    """
    L-layer backward propagation with configurable options.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    use_dropout = keep_prob < 1.0
    use_batchnorm = bn_params is not None
    
    # Output layer gradient
    linear_cache, activation_cache, _, _ = caches[L - 1]
    
    epsilon = 1e-15  # Prevent divide by zero
    if output == "softmax":
        dZ = AL - Y
    elif output == "linear":
        # MSE gradient: dZ = (AL - Y) / m (but we divide by m later in linear backward)
        dZ = AL - Y
    else:  # sigmoid
        AL_clipped = np.clip(AL, epsilon, 1 - epsilon)
        dAL = -(np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped))
        dZ = sigmoid_backward(dAL, activation_cache)
    
    A_prev, W, b = linear_cache
    grads["dW" + str(L)] = 1/m * np.dot(dZ, A_prev.T)
    if lambd > 0:
        grads["dW" + str(L)] += (lambd / m) * parameters['W' + str(L)]
    grads["db" + str(L)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    grads["dA" + str(L-1)] = np.dot(W.T, dZ)
    
    # Hidden layers (reverse order)
    for l in reversed(range(L - 1)):
        linear_cache, activation_cache, bn_cache, D = caches[l]
        
        dA = grads["dA" + str(l + 1)]
        
        # Dropout backward
        # Dropout backward
        if use_dropout and D is not None:
             dA = dropout_backward(dA, D) # D here is the cache tuple (D_mat, keep_prob) stored in forward loop
        
        # ReLU backward
        dZ = relu_backward(dA, activation_cache)
        
        # Batch norm backward
        if use_batchnorm and bn_cache is not None:
            dZ, dgamma, dbeta = batchnorm_backward(dZ, bn_cache)
            grads["dgamma" + str(l + 1)] = dgamma
            grads["dbeta" + str(l + 1)] = dbeta
        
        # Linear backward with optional L2
        A_prev, W, b = linear_cache
        grads["dW" + str(l + 1)] = 1/m * np.dot(dZ, A_prev.T)
        if lambd > 0:
            grads["dW" + str(l + 1)] += (lambd / m) * parameters['W' + str(l + 1)]
        grads["db" + str(l + 1)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        grads["dA" + str(l)] = np.dot(W.T, dZ)
    
    return grads


# =============================================================================
# CNN Layers (Convolution, Pooling, Flatten)
# =============================================================================

def zero_pad(X, pad):
    """
    Pad all images in  Xwith zeros around height and width.
    
    Arguments:
    X -- numpy array of shape (m, n_H, n_W, n_C)
    pad -- integer, amount of padding around each image
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 
                   mode='constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter on a single slice of the input.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters - matrix of shape (f, f, n_C_prev)
    b -- Bias parameter - scalar
    
    Returns:
    Z -- scalar, result of convolving filter W on slice
    """
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + float(b)
    return Z


# =============================================================================
# Helper Functions for Vectorized Convolution (im2col/col2im)
# =============================================================================

from .utils import get_im2col_indices, im2col_indices, col2im_indices


# =============================================================================
# Vectorized Convolution Functions (using im2col)
# =============================================================================

def conv_forward(A_prev, W, b, hparameters):
    """
    Forward propagation for a convolution layer (Vectorized).
    
    Arguments:
    A_prev -- output of previous layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, shape (f, f, n_C_prev, n_C)
    b -- Biases, shape (1, 1, 1, n_C)
    hparameters -- dict containing "stride" and "pad"
        
    Returns:
    Z -- conv output, shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for conv_backward()
    """
    # Retrieve dimensions
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Calculate output dimensions
    n_H = int(((n_H_prev - f + (2 * pad)) / stride)) + 1
    n_W = int(((n_W_prev - f + (2 * pad)) / stride)) + 1
    
    # Transpose A_prev to (m, n_C_prev, n_H_prev, n_W_prev) for im2col helpers
    A_prev_T = A_prev.transpose(0, 3, 1, 2)
    
    # Create columns: Shape (f*f*n_C_prev, m*n_H*n_W)
    X_col = im2col_indices(A_prev_T, f, f, padding=pad, stride=stride)
    
    # Flatten weights: Shape (n_C, f*f*n_C_prev)
    # W shape is (f, f, n_C_prev, n_C) -> transpose to (n_C, n_C_prev, f, f) -> reshape
    W_col = W.transpose(3, 2, 0, 1).reshape(n_C, -1)
    
    # Perform matrix multiplication
    # Z_col shape: (n_C, m*n_H*n_W)
    Z_col = W_col @ X_col + b.reshape(-1, 1)
    
    # Reshape back to image: (n_C, n_H, n_W, m) -> transpose to (m, n_H, n_W, n_C)
    Z = Z_col.reshape(n_C, n_H, n_W, m).transpose(3, 1, 2, 0)
    
    cache = (A_prev, W, b, hparameters, X_col)
    return Z, cache


def conv_backward(dZ, cache):
    """
    Backward propagation for a convolution layer (Vectorized).
    """
    (A_prev, W, b, hparameters, X_col) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Gradient of bias: sum over (m, n_H, n_W)
    db = np.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, n_C)
    
    # Reshape dZ: (n_C, n_H, n_W, m) -> flatten to (n_C, m*n_H*n_W)
    dZ_col = dZ.transpose(3, 1, 2, 0).reshape(n_C, -1)
    
    # Gradient of weights: dW = dZ_col @ X_col.T
    dW_col = dZ_col @ X_col.T
    # Reshape dW_col (n_C, f*f*n_C_prev) -> (n_C, n_C_prev, f, f) -> (f, f, n_C_prev, n_C)
    dW = dW_col.reshape(n_C, n_C_prev, f, f).transpose(2, 3, 1, 0)
    
    # Gradient of input
    W_col = W.transpose(3, 2, 0, 1).reshape(n_C, -1)
    dX_col = W_col.T @ dZ_col  # (f*f*n_C_prev, m*n_H*n_W)
    
    # Col2im: (m, n_C_prev, n_H_prev, n_W_prev)
    dA_prev_T = col2im_indices(dX_col, (m, n_C_prev, n_H_prev, n_W_prev), 
                               field_height=f, field_width=f, padding=pad, stride=stride)
    
    # Transpose back to (m, n_H_prev, n_W_prev, n_C_prev)
    dA_prev = dA_prev_T.transpose(0, 2, 3, 1)
    
    return dA_prev, dW, db


def conv_forward_naive(A_prev, W, b, hparameters):
    """
    Forward propagation for a convolution layer.
    
    Arguments:
    A_prev -- output of previous layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, shape (f, f, n_C_prev, n_C)
    b -- Biases, shape (1, 1, 1, n_C)
    hparameters -- dict containing "stride" and "pad"
        
    Returns:
    Z -- conv output, shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for conv_backward()
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f2, n_C_prev2, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int(((n_H_prev - f + (2 * pad)) / stride)) + 1
    n_W = int(((n_W_prev - f + (2 * pad)) / stride)) + 1
    
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def conv_backward_naive(dZ, cache):
    """
    Backward propagation for a convolution layer.
    
    Arguments:
    dZ -- gradient of cost w.r.t. conv output, shape (m, n_H, n_W, n_C)
    cache -- cache from conv_forward()
    
    Returns:
    dA_prev -- gradient w.r.t. input, shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient w.r.t. weights, shape (f, f, n_C_prev, n_C)
    db -- gradient w.r.t. biases, shape (1, 1, 1, n_C)
    """
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f2, n_C_prev2, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]
        
        if pad > 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad
    
    return dA_prev, dW, db


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Forward pass of the pooling layer (Vectorized).
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    
    # Reshape to (m*n_C, 1, n_H, n_W) to process each channel independently
    A_prev_reshaped = A_prev.transpose(0, 3, 1, 2).reshape(m * n_C_prev, 1, n_H_prev, n_W_prev)
    
    # Use im2col to extract windows
    cols = im2col_indices(A_prev_reshaped, f, f, padding=0, stride=stride)
    
    if mode == "max":
        A_pool_col = np.max(cols, axis=0)
        # Store argmax indices for backward pass
        arg_max = np.argmax(cols, axis=0)
    elif mode == "average":
        A_pool_col = np.mean(cols, axis=0)
        arg_max = None
        
    # Reshape result
    # A_pool_col layout: (out_h, out_w, m, n_C) due to im2col being (pixels, spatial*batch)
    # where batch varies fastest.
    # N here is m*n_C. So it is (out_h, out_w, m*n_C).
    # Reshape to (n_H, n_W, m, n_C)
    A_pool = A_pool_col.reshape(n_H, n_W, m, n_C_prev).transpose(2, 0, 1, 3)
    
    cache = (A_prev, A_prev_reshaped, hparameters, arg_max, cols.shape)
    return A_pool, cache


def pool_forward_naive(A_prev, hparameters, mode="max"):
    """
    Forward pass of the pooling layer (Naive loops).
    
    Arguments:
    A_prev -- Input data, shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- dict containing "f" (pool size) and "stride"
    mode -- "max" or "average"
    
    Returns:
    A -- output of pool layer, shape (m, n_H, n_W, n_C)
    cache -- cache for backward pass
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    cache = (A_prev, hparameters)
    return A, cache


def create_mask_from_window(x):
    """Create a mask identifying the max entry of x."""
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    """Distribute value evenly across a matrix (for average pooling backward)."""
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones((n_H, n_W)) * average
    return a


def pool_backward(dA, cache, mode="max"):
    """
    Backward pass of the pooling layer (Vectorized).
    """
    (A_prev, A_prev_reshaped, hparameters, arg_max, cols_shape) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Flatten dA to match columns
    # cols order is [p0_image0_c0, p0_image0_c1... p0_image1_c0...] (spatial, batch, channel)
    # Actually im2col flattened batch*channel into N.
    # cols has N varying fastest. N = m*n_C.
    # So we need dA to be (n_H, n_W, m, n_C) -> flatten
    dA_flat = dA.transpose(1, 2, 0, 3).flatten()
    
    dA_cols = np.zeros(cols_shape)
    
    if mode == "max":
        # Assign gradients to max indices
        n_cols = cols_shape[1]
        dA_cols[arg_max, np.arange(n_cols)] = dA_flat
        
    elif mode == "average":
        # Distribute gradient equally
        dA_cols = np.tile(dA_flat / (f*f), (cols_shape[0], 1))
    
    # Col2im: map columns back to image
    # A_prev_reshaped was (m*n_C, 1, n_H_prev, n_W_prev)
    shape_reshaped = A_prev_reshaped.shape 
    dA_prev_reshaped = col2im_indices(dA_cols, shape_reshaped, f, f, padding=0, stride=stride)
    
    # Reshape back to original format
    # dA_prev_reshaped: (m*n_C, 1, n_H_prev, n_W_prev) -> (m, n_C, n_H_prev, n_W_prev) -> (m, n_H_prev, n_W_prev, n_C)
    dA_prev = dA_prev_reshaped.reshape(m, n_C_prev, n_H_prev, n_W_prev).transpose(0, 2, 3, 1)
    
    return dA_prev


def pool_backward_naive(dA, cache, mode="max"):
    """
    Backward pass of the pooling layer (Naive loops).
    
    Arguments:
    dA -- gradient w.r.t. pool output, shape (m, n_H, n_W, n_C)
    cache -- cache from pool_forward()
    mode -- "max" or "average"
    
    Returns:
    dA_prev -- gradient w.r.t. pool input, shape (m, n_H_prev, n_W_prev, n_C_prev)
    """
    A_prev, hparameters = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros(A_prev.shape)
    
    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
    
    return dA_prev


def flatten_forward(A):
    """
    Flatten 4D tensor to 2D for dense layers.
    
    Arguments:
    A -- input of shape (m, n_H, n_W, n_C)
    
    Returns:
    A_flat -- flattened output of shape (n_H * n_W * n_C, m)
    cache -- original shape for backward pass
    """
    m = A.shape[0]
    A_flat = A.reshape(m, -1).T
    cache = A.shape
    return A_flat, cache


def flatten_backward(dA_flat, cache):
    """
    Reshape gradients from dense layer back to conv format.
    
    Arguments:
    dA_flat -- gradient from dense layer, shape (features, m)
    cache -- original 4D shape (m, n_H, n_W, n_C)
    
    Returns:
    dA -- reshaped gradient, shape (m, n_H, n_W, n_C)
    """
    original_shape = cache
    dA = dA_flat.T.reshape(original_shape)
    return dA


# =============================================================================
# Residual Block (ResNet Skip Connection)
# =============================================================================

def residual_forward(A_input, parameters, res_idx, layer_config):
    """
    Forward propagation for a residual block.
    
    Structure: Conv1 -> ReLU -> Conv2 + skip -> ReLU
    
    Arguments:
    A_input -- input activation, shape (m, n_H, n_W, n_C_prev)
    parameters -- dict containing W and b for this residual block
    res_idx -- index of this residual block
    layer_config -- dict with 'kernel_size', 'downsample', 'has_skip_conv'
    
    Returns:
    A_out -- output activation after residual block
    cache -- dict with all caches needed for backward pass
    """
    f = layer_config["kernel_size"]
    downsample = layer_config["downsample"]
    stride1 = 2 if downsample else 1
    pad = f // 2
    
    # Main path: Conv1 -> ReLU -> Conv2
    W1 = parameters[f"W_res{res_idx}_conv1"]
    b1 = parameters[f"b_res{res_idx}_conv1"]
    hparams1 = {"stride": stride1, "pad": pad}
    Z1, cache_conv1 = conv_forward(A_input, W1, b1, hparams1)
    A1, cache_relu1 = relu(Z1)
    
    W2 = parameters[f"W_res{res_idx}_conv2"]
    b2 = parameters[f"b_res{res_idx}_conv2"]
    hparams2 = {"stride": 1, "pad": pad}
    Z2, cache_conv2 = conv_forward(A1, W2, b2, hparams2)
    
    # Skip path
    if layer_config.get("has_skip_conv", False):
        W_skip = parameters[f"W_res{res_idx}_skip"]
        b_skip = parameters[f"b_res{res_idx}_skip"]
        hparams_skip = {"stride": stride1, "pad": 0}
        A_skip, cache_skip = conv_forward(A_input, W_skip, b_skip, hparams_skip)
    else:
        A_skip = A_input
        cache_skip = None
    
    # Add skip connection and apply final ReLU
    Z_out = Z2 + A_skip
    A_out, cache_relu_out = relu(Z_out)
    
    cache = {
        "cache_conv1": cache_conv1,
        "cache_relu1": cache_relu1,
        "cache_conv2": cache_conv2,
        "cache_skip": cache_skip,
        "cache_relu_out": cache_relu_out,
        "A_input": A_input,
        "has_skip_conv": layer_config.get("has_skip_conv", False)
    }
    
    return A_out, cache


def residual_backward(dA, cache, res_idx):
    """
    Backward propagation for a residual block.
    
    Gradients flow through both main path and skip path, then combine.
    
    Arguments:
    dA -- gradient w.r.t. block output, shape (m, n_H, n_W, n_C)
    cache -- dict from residual_forward
    res_idx -- index of this residual block
    
    Returns:
    dA_input -- gradient w.r.t. block input
    grads -- dict with gradients for all parameters in this block
    """
    grads = {}
    
    # Backward through final ReLU
    dZ_out = relu_backward(dA, cache["cache_relu_out"])
    
    # Gradient splits to both paths
    dZ2 = dZ_out  # Main path gradient
    dA_skip = dZ_out  # Skip path gradient
    
    # Main path backward: Conv2 -> ReLU1 -> Conv1
    dA1, dW2, db2 = conv_backward(dZ2, cache["cache_conv2"])
    grads[f"dW_res{res_idx}_conv2"] = dW2
    grads[f"db_res{res_idx}_conv2"] = db2
    
    dZ1 = relu_backward(dA1, cache["cache_relu1"])
    dA_input_main, dW1, db1 = conv_backward(dZ1, cache["cache_conv1"])
    grads[f"dW_res{res_idx}_conv1"] = dW1
    grads[f"db_res{res_idx}_conv1"] = db1
    
    # Skip path backward
    if cache["has_skip_conv"]:
        dA_input_skip, dW_skip, db_skip = conv_backward(dA_skip, cache["cache_skip"])
        grads[f"dW_res{res_idx}_skip"] = dW_skip
        grads[f"db_res{res_idx}_skip"] = db_skip
    else:
        dA_input_skip = dA_skip
    
    # Combine gradients from both paths
    dA_input = dA_input_main + dA_input_skip
    
    return dA_input, grads


# =============================================================================
# CNN Model Forward/Backward (Full Network)
# =============================================================================





def CNN_model_forward(X, layers, parameters, keep_prob=1.0, training=True,
                      bn_params=None, bn_running=None):
    """
    Forward propagation for the entire CNN.
    
    Arguments:
    X -- Input data, shape (m, n_H, n_W, n_C)
    layers -- List of layer configurations
    parameters -- Dictionary containing parameters
    keep_prob -- Dropout keep probability (1.0 = no dropout)
    training -- Whether in training mode (dropout only applied if True)
    bn_params -- Batch norm parameters (gamma, beta) for conv layers (optional)
    bn_running -- Batch norm running stats for inference (optional)
    
    Returns:
    AL -- Output of the last layer
    caches -- list of caches from each layer
    """
    caches = []
    A = X
    L = len(layers)
    use_dropout = keep_prob < 1.0 and training
    use_batchnorm = bn_params is not None
    conv_idx = 0
    dense_idx = 0
    
    # Loop over layers
    for l, layer in enumerate(layers):
        layer_type = layer["type"]
        
        if layer_type == "conv":
            conv_idx += 1
            W = parameters[f'W_conv{conv_idx}']
            b = parameters[f'b_conv{conv_idx}']
            hparameters = {"stride": layer["stride"], "pad": layer["pad"]}
            A, cache = conv_forward(A, W, b, hparameters)
            caches.append((layer_type, cache, conv_idx))
            
            # Batch Normalization BEFORE activation (Conv -> BN -> ReLU)
            if use_batchnorm:
                # Get shape: (m, n_H, n_W, n_C)
                m_batch, n_H, n_W, n_C = A.shape
                
                # Reshape: (m, H, W, C) -> (C, m*H*W)
                A_reshaped = A.transpose(3, 0, 1, 2).reshape(n_C, -1)
                
                # Get gamma, beta for this conv layer
                gamma = bn_params[f'gamma_conv{conv_idx}']
                beta = bn_params[f'beta_conv{conv_idx}']
                
                # Apply batch norm (reusing existing function)
                A_norm, bn_cache = batchnorm_forward(
                    A_reshaped, gamma, beta,
                    bn_running=bn_running,
                    layer_idx=f'_conv{conv_idx}',  # Use string key for CNN
                    training=training
                )
                
                # Reshape back: (C, m*H*W) -> (m, H, W, C)
                A = A_norm.reshape(n_C, m_batch, n_H, n_W).transpose(1, 2, 3, 0)
                
                # Store BN cache with shape info for backward pass
                caches.append(("bn_conv", bn_cache, conv_idx, (m_batch, n_H, n_W, n_C)))
            
            # Activation AFTER batch norm
            if layer["activation"] == "relu":
                A, activation_cache = relu(A)
                caches.append(("relu", activation_cache))
                
        elif layer_type == "pool":
            hparameters = {"f": layer["pool_size"], "stride": layer["stride"]}
            mode = layer["mode"]
            A, cache = pool_forward(A, hparameters, mode=mode)
            caches.append((layer_type, cache))
            
        elif layer_type == "flatten":
            A, cache = flatten_forward(A)
            caches.append((layer_type, cache))
            
        elif layer_type == "dense":
            dense_idx += 1
            W = parameters[f'W_dense{dense_idx}']
            b = parameters[f'b_dense{dense_idx}']
            A, linear_cache = linear_forward(A, W, b)
            
            if layer["activation"] == "relu":
                A, activation_cache = relu(A)
                cache = (linear_cache, activation_cache, dense_idx)
                caches.append(("dense_relu", cache))
                
                # Apply dropout after dense relu (not output layer)
                if use_dropout:
                    A, D_cache = dropout_forward(A, keep_prob)
                    caches.append(("dropout", D_cache))
                    
            elif layer["activation"] == "softmax":
                A, activation_cache = softmax(A)
                cache = (linear_cache, activation_cache, dense_idx)
                caches.append(("dense_softmax", cache))
                
        elif layer_type == "residual":
            # Count residual blocks so far
            res_idx = sum(1 for i2, l2 in enumerate(layers[:l+1]) if l2["type"] == "residual")
            
            # Use standalone residual_forward function
            A, cache = residual_forward(A, parameters, res_idx, layer)
            cache["res_idx"] = res_idx  # Ensure res_idx is in cache for backward
            caches.append((layer_type, cache))
            
        elif layer_type == "dropout":
             keep_prob = layer["keep_prob"]
             A, cache = dropout_forward(A, keep_prob)
             caches.append((layer_type, cache))

    return A, caches


def CNN_model_backward(AL, Y, layers, caches, keep_prob=1.0, use_batchnorm=False):
    """
    Backward propagation for the entire CNN.
    
    Arguments:
    AL -- Output of forward prop
    Y -- True labels
    layers -- Layer configurations
    caches -- Caches from forward prop
    keep_prob -- Dropout keep probability. If < 1.0, expects dropout caches after dense_relu.
    use_batchnorm -- Whether batch normalization was used in forward pass
    """
    grads = {}
    L = len(layers)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    use_dropout = keep_prob < 1.0
    
    # Derivative of cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    dA_prev = dAL
    
    # Loop over layers backward
    for l in reversed(range(L)):
        layer = layers[l]
        layer_type = layer["type"]
        
        if layer_type == "dense":
            # Check if there's a dropout cache to pop first (for relu layers)
            if use_dropout and layer["activation"] == "relu":
                # Pop dropout cache
                dropout_entry = caches.pop()
                if dropout_entry[0] == "dropout":
                    dA_prev = dropout_backward(dA_prev, dropout_entry[1])
                else:
                    # Not a dropout cache, put it back (shouldn't happen if forward is correct)
                    caches.append(dropout_entry)
            
            cache_entry = caches.pop()
            type_tag = cache_entry[0]
            # Cache now has 3 elements: (type_tag, (linear_cache, activation_cache), dense_idx)
            linear_cache, activation_cache, dense_idx = cache_entry[1], cache_entry[2] if len(cache_entry) > 2 else None, None
            # Actually cache_entry is (type_tag, (linear_cache, activation_cache, dense_idx))
            # Let me check forward: cache = (linear_cache, activation_cache, dense_idx)
            # So cache_entry = ("dense_relu", (linear_cache, activation_cache, dense_idx))
            cache_data = cache_entry[1]
            linear_cache = cache_data[0]
            activation_cache = cache_data[1]
            dense_idx = cache_data[2]
            
            if "softmax" in layer["activation"]:
                dZ = AL - Y
                A_prev, W, b = linear_cache
                m = A_prev.shape[1]
                dW = 1./m * np.dot(dZ, A_prev.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims=True)
                dA_prev = np.dot(W.T, dZ)
                
                grads[f'dW_dense{dense_idx}'] = dW
                grads[f'db_dense{dense_idx}'] = db
                
            elif "relu" in layer["activation"]:
                # Inline relu backward: dZ = dA * relu'(Z)
                dZ = dA_prev * (activation_cache > 0)
                # Inline linear backward
                A_prev, W, b = linear_cache
                m = A_prev.shape[1]
                dW = 1./m * np.dot(dZ, A_prev.T)
                db = 1./m * np.sum(dZ, axis=1, keepdims=True)
                dA_prev = np.dot(W.T, dZ)
                
                grads[f'dW_dense{dense_idx}'] = dW
                grads[f'db_dense{dense_idx}'] = db
                
        elif layer_type == "flatten":
            cache_entry = caches.pop()
            type_tag, cache_val = cache_entry
            dA_prev = flatten_backward(dA_prev, cache_val)
            
        elif layer_type == "pool":
            cache_entry = caches.pop()
            type_tag, cache_val = cache_entry
            mode = layer["mode"]
            dA_prev = pool_backward(dA_prev, cache_val, mode=mode)
            
        elif layer_type == "conv":
            # Backward order: ReLU -> BN -> Conv (reverse of forward: Conv -> BN -> ReLU)
            
            # 1. Pop relu first
            relu_entry = caches.pop()  # ("relu", cache)
            dA_prev = relu_backward(dA_prev, relu_entry[1])
            
            # 2. Then pop and process batch norm if enabled
            if use_batchnorm:
                bn_entry = caches.pop()
                if bn_entry[0] == "bn_conv":
                    # bn_entry = ("bn_conv", bn_cache, conv_idx, (m, n_H, n_W, n_C))
                    bn_cache = bn_entry[1]
                    conv_idx = bn_entry[2]
                    orig_shape = bn_entry[3]  # (m, n_H, n_W, n_C)
                    m_batch, n_H, n_W, n_C = orig_shape
                    
                    # Reshape dA_prev: (m, H, W, C) -> (C, m*H*W)
                    dA_reshaped = dA_prev.transpose(3, 0, 1, 2).reshape(n_C, -1)
                    
                    # Apply batch norm backward
                    dA_norm, dgamma, dbeta = batchnorm_backward(dA_reshaped, bn_cache)
                    
                    # Store BN gradients
                    grads[f'dgamma_conv{conv_idx}'] = dgamma
                    grads[f'dbeta_conv{conv_idx}'] = dbeta
                    
                    # Reshape back: (C, m*H*W) -> (m, H, W, C)
                    dA_prev = dA_norm.reshape(n_C, m_batch, n_H, n_W).transpose(1, 2, 3, 0)
                else:
                    # Not a BN cache, put it back
                    caches.append(bn_entry)
            
            # 3. Finally pop and process conv
            conv_entry = caches.pop()  # ("conv", cache, conv_idx)
            conv_cache = conv_entry[1]
            conv_idx = conv_entry[2]
            dA_prev, dW, db = conv_backward(dA_prev, conv_cache)
            
            grads[f'dW_conv{conv_idx}'] = dW
            grads[f'db_conv{conv_idx}'] = db
            
        elif layer_type == "dropout":
            cache_entry = caches.pop()
            type_tag, cache_val = cache_entry
            dA_prev = dropout_backward(dA_prev, cache_val)
            
        elif layer_type == "residual":
            cache_entry = caches.pop()
            type_tag, cache = cache_entry
            res_idx = cache["res_idx"]
            
            # Use standalone residual_backward function
            dA_prev, res_grads = residual_backward(dA_prev, cache, res_idx)
            
            # Merge residual gradients into main grads dict
            grads.update(res_grads)

    return grads

