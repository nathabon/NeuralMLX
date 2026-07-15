import cupy as cp
import numpy as np
from typing import Any

# ─── Types ────────────────────────────────────────────────────────────────────

float32 = cp.float32
float16 = cp.float16
int32   = cp.int32
bool_   = cp.bool_
uint8   = cp.uint8

# ─── Constructeurs ────────────────────────────────────────────────────────────

def array(data, dtype=None) -> cp.ndarray:
    if isinstance(data, cp.ndarray):
        return data.astype(dtype) if dtype is not None else data
    return cp.array(data, dtype=dtype)

def zeros(shape, dtype=float32) -> cp.ndarray:
    return cp.zeros(shape, dtype=dtype)

def ones(shape, dtype=float32) -> cp.ndarray:
    return cp.ones(shape, dtype=dtype)

def zeros_like(x: cp.ndarray) -> cp.ndarray:
    return cp.zeros_like(x)

def ones_like(x: cp.ndarray) -> cp.ndarray:
    return cp.ones_like(x)

def eye(n: int, dtype=float32) -> cp.ndarray:
    return cp.eye(n, dtype=dtype)

def arange(*args, **kwargs) -> cp.ndarray:
    return cp.arange(*args, **kwargs)

# ─── Maths élémentaires ───────────────────────────────────────────────────────

def exp(x):      return cp.exp(x)
def log(x):      return cp.log(x)
def sqrt(x):     return cp.sqrt(x)
def abs(x):      return cp.abs(x)
def maximum(a, b): return cp.maximum(a, b)
def minimum(a, b): return cp.minimum(a, b)
def clip(x, a, b): return cp.clip(x, a, b)

# ─── Réductions ───────────────────────────────────────────────────────────────

def sum(x, axis=None, keepdims=False):
    return cp.sum(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    return cp.mean(x, axis=axis, keepdims=keepdims)

def max(x, axis=None, keepdims=False):
    return cp.max(x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    return cp.min(x, axis=axis, keepdims=keepdims)

def argmax(x, axis=None):
    return cp.argmax(x, axis=axis)

# ─── Manipulation de shape ────────────────────────────────────────────────────

def transpose(x, axes=None):
    return cp.transpose(x, axes)

def reshape(x, shape):
    return cp.reshape(x, shape)

def concatenate(arrays, axis=0):
    return cp.concatenate(arrays, axis=axis)

def stack(arrays, axis=0):
    return cp.stack(arrays, axis=axis)

def broadcast_to(x, shape):
    return cp.broadcast_to(x, shape)

def pad(x, pad_width, mode="constant", **kwargs):
    return cp.pad(x, pad_width, mode=mode, **kwargs)

def flatten(x):
    return x.flatten()

# ─── Algèbre linéaire ─────────────────────────────────────────────────────────

def matmul(a, b):
    return cp.matmul(a, b)

# ─── Random ───────────────────────────────────────────────────────────────────

class random:
    @staticmethod
    def uniform(low=0.0, high=1.0, shape=None):
        return cp.random.uniform(low, high, shape).astype(cp.float32)

    @staticmethod
    def normal(shape=None):
        return cp.random.normal(size=shape).astype(cp.float32)

    @staticmethod
    def randint(low, high=None, shape=None):
        return cp.random.randint(low, high, size=shape)


# ─── Conv2d (im2col + matmul sur GPU) ─────────────────────────────────────────
#
# MLX fournit mx.conv2d(input, kernel, stride) avec la convention :
#   input  : (B, H,   W,   Cin)
#   kernel : (Cout, kH, kW, Cin)
#   sortie : (B, H_out, W_out, Cout)
#
# On reproduit exactement cette convention avec CuPy via im2col.

def _im2col_strided(X: cp.ndarray, kH: int, kW: int, stride: int = 1) -> cp.ndarray:
    """
    (B, H, W, C) → (B, H_out*W_out, kH*kW*C)
    Boucle Python sur kH×kW uniquement — tout le reste est vectorisé CuPy.
    """
    B, H, W, C = X.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    cols = []
    for dh in range(kH):
        for dw in range(kW):
            p = X[:,
                  dh : dh + H_out * stride : stride,
                  dw : dw + W_out * stride : stride,
                  :]                                    # (B, H_out, W_out, C)
            cols.append(p)

    return cp.concatenate(cols, axis=-1).reshape(B, H_out * W_out, kH * kW * C)


def conv2d(X, kernel, stride=1):
    """
    Convolution 2D pure CuPy, sans aller-retour GPU→CPU.

    Convention identique à MLX :
      X      : (B, H, W, Cin)
      kernel : (Cout, kH, kW, Cin)
      retour : (B, H_out, W_out, Cout)
    """
    X = cp.asarray(X)
    kernel = cp.asarray(kernel)

    B, H, W, Cin = X.shape
    Cout, kH, kW, Cin_k = kernel.shape
    if Cin != Cin_k:
        raise ValueError(f"conv2d: Cin mismatch: X has {Cin}, kernel has {Cin_k}")

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # (B, H_out*W_out, kH*kW*Cin)
    X_col = _im2col_strided(X, kH, kW, stride)

    # (kH*kW*Cin, Cout)
    K_col = kernel.reshape(Cout, -1).T

    # (B, H_out*W_out, Cout)
    out = cp.matmul(X_col, K_col)
    return out.reshape(B, H_out, W_out, Cout)


def conv_transpose2d(delta: cp.ndarray, kernel_t: cp.ndarray, stride: int = 1) -> cp.ndarray:
    """
    Convolution transposée — gradient vers l'entrée d'une conv2d.

    delta    : (B, H_out, W_out, Cout)
    kernel_t : (Cin, kH, kW, Cout)   — kernel transposé (dims in/out échangées)
    retour   : (B, H_in,  W_in,  Cin)  approximatif (corrigé par match_spatial_shape)

    Implémentation : dilatation de delta (insertion stride-1 zéros) + conv valide.
    Pour stride=1, c'est simplement une conv valide classique.
    """
    B, H_out, W_out, Cout = delta.shape
    Cin, kH, kW, _        = kernel_t.shape

    if stride == 1:
        # Conv valide directe
        # Reformuler : delta comme input, kernel_t comme filtre
        # (B, H_out, W_out, Cout) * (Cin, kH, kW, Cout) → (B, H_in, W_in, Cin)
        # Astuce : traiter Cout comme batch de filtres
        H_in = H_out + kH - 1
        W_in = W_out + kW - 1

        # Padding "full" de delta
        delta_pad = cp.pad(
            delta,
            [(0,0), (kH-1, kH-1), (kW-1, kW-1), (0,0)]
        )                                                  # (B, H_in+kH-1, W_in+kH-1, Cout)

        # kernel_t retourné (flip spatial) pour la conv corrélation→convolution
        k_flip = kernel_t[:, ::-1, ::-1, :]               # (Cin, kH, kW, Cout)

        # Appel récursif conv2d stride=1
        # input : (B, H_pad, W_pad, Cout)
        # kernel: (Cin, kH, kW, Cout) — Cout joue le rôle de Cin
        out = conv2d(delta_pad, k_flip, stride=1)          # (B, H_in, W_in, Cin)
        return out

    else:
        # Dilatation de delta : insérer (stride-1) zéros entre chaque élément
        H_dil = H_out + (H_out - 1) * (stride - 1)
        W_dil = W_out + (W_out - 1) * (stride - 1)
        delta_dil = cp.zeros((B, H_dil, W_dil, Cout), dtype=delta.dtype)
        delta_dil[:, ::stride, ::stride, :] = delta

        # Puis conv valide classique (stride=1) avec padding full
        return conv_transpose2d(delta_dil, kernel_t, stride=1)


# ─── Lazy eval — no-op sous CuPy (exécution immédiate) ───────────────────────

def eval(*args):
    """
    MLX est lazy : mx.eval() force l'exécution du graphe.
    CuPy est eager (comme NumPy) : les opérations s'exécutent immédiatement.
    Cette fonction est un no-op, présente uniquement pour la compatibilité.
    """
    pass

def synchronize():
    """Attend la fin des kernels CUDA en cours."""
    cp.cuda.Stream.null.synchronize()


# ─── Utilitaire : transfert GPU ↔ CPU ─────────────────────────────────────────

def to_numpy(x) -> np.ndarray:
    """Transfère un array CuPy vers NumPy (CPU)."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

def to_gpu(x, dtype=None) -> cp.ndarray:
    """Transfère un array NumPy vers CuPy (GPU)."""
    return cp.array(x, dtype=dtype)
