import sys
if sys.platform != "darwin":
    raise ImportError("This file is only for macOS (darwin) platform.")
import mlx.core as mx
from collections.abc import Callable

def fx(X) -> mx.array:
    return X

def xPrime(X):
    return mx.ones(X.shape)

def sigmoid(x):
    return 1 / (1 + mx.exp(-x))

def sigmoidPrime(x):
    s = sigmoid(x)
    return s * (1 - s)

def ReLU(x):
    return mx.maximum(x, 0)

def ReLUPrime(x):
    return (x > 0).astype(mx.float32)

def normalize(x: mx.array):
    s = x.sum(axis=-1, keepdims=True)
    return x / s

def normalizePrime(x: mx.array):
    s = x.sum(axis=-1, keepdims=True)
    return (s - x) / (s ** 2)

def softmax(x):
    x = x - mx.max(x, axis=1, keepdims=True)
    exp = mx.exp(x)
    return exp / mx.sum(exp, axis=1, keepdims=True)

def softmaxPrime(x: mx.array):
    s = softmax(x)
    return s * (1 - s)



ACTIVATIONS: dict[str, Callable] = {
    "fx": fx,
    "sigmoid": sigmoid,
    "relu": ReLU,
    "normalize": normalize,
    "softmax": softmax
}

def prime(func: Callable) -> Callable: # type: ignore
    if func == fx:
        return xPrime
    elif func == sigmoid:
        return sigmoidPrime
    elif func == ReLU:
        return ReLUPrime
    elif func == softmax:
        return softmaxPrime
    
def zeros_hot(size, index, nb=1.):
    data = [0.] * size
    data[index] = float(nb)
    return mx.array(data)


def im2col(X: mx.array, kH: int, kW: int):
    B, H, W, Cin = X.shape

    Hout = H - kH + 1
    Wout = W - kW + 1

    patches = []

    for i in range(Hout):
        for j in range(Wout):
            patch = X[:, i:i+kH, j:j+kW, :]

            patches.append(
                patch.reshape(B, -1)
            )

    return mx.stack(patches, axis=1)

def im2col_strided(X, kH, kW, stride=1):
    B, H, W, C = X.shape

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    cols = []

    for i in range(H_out):
        for j in range(W_out):
            y = i * stride
            x = j * stride
            patch = X[:, y:y+kH, x:x+kW, :]          # (B, kH, kW, C)
            cols.append(patch.reshape(B, -1))        # (B, kH*kW*C)

    return mx.stack(cols, axis=1)  # (B, H_out*W_out, kH*kW*C)

def match_spatial_shape(x: mx.array, target_shape: tuple) -> mx.array:
    _, target_H, target_W, _ = target_shape

    # Crop si trop grand
    x = x[:, :target_H, :target_W, :]

    # Pad si trop petit
    pad_H = target_H - x.shape[1]
    pad_W = target_W - x.shape[2]

    if pad_H > 0 or pad_W > 0:
        x = mx.pad(
            x,
            [(0, 0), (0, pad_H), (0, pad_W), (0, 0)]
        )

    return x