from . import backend_cupy as mx
import numpy as np
from collections.abc import Callable
from .other import *
import h5py

def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def no_grad(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        old_training = getattr(self, 'training', True)
        self.training = False
        result = func(*args, **kwargs)
        self.training = old_training
        return result
    return wrapper

def train(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        old_training = getattr(self, 'training', True)
        self.training = True
        result = func(*args, **kwargs)
        self.training = old_training
        return result
    return wrapper

#MARK: Layer
class Layer:
    weights: mx.array
    grad: mx.array
    training: bool
    last_X: mx.array
    last_Z: mx.array

    def __init__(self, *args, **kwargs):
        self.training = True
        self.grad = None # type: ignore
        self.last_X = None # type: ignore
        self.last_Z = None # type: ignore
        self.weights = None # type: ignore
        self.__dict__.update(kwargs)
        
    
    @classmethod
    def fromH5(cls, grp: h5py.Group) -> "Layer":
        typ = grp.attrs["type"]
        match typ:
            case "NeuralLayer":
                return NeuralLayer.fromH5(grp)
            
            case "ConvolutionalLayer":
                return ConvolutionalLayer.fromH5(grp)
            
            case "PoolingLayer":
                return PoolingLayer.fromH5(grp)
            
            case "FlattenLayer":
                return FlattenLayer.fromH5(grp)
            
            case _:
                raise NotImplementedError(f"fromH5 n'est pas implémenté pour le type {typ}")
    
    def copy(self) -> 'Layer':
        raise NotImplementedError(f"{type(self).__name__} doit implémenter copy")
    
    def copy_from(self, other: 'Layer'):
        raise NotImplementedError(f"{type(self).__name__} doit implémenter copy_from")

    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.__dict__}>"

    def __repr__(self) -> str:
        return str(self)

    def __call__(self, X: mx.array) -> mx.array:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter __call__")

    def backward(self, delta: mx.array) -> mx.array:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter backward")

    def update(self, learningRate: float):
        if self.weights is not None and self.grad is not None:
            self.weights = self.weights - learningRate * self.grad
    
    def getNbParameters(self) -> int:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter getNbParameters")
    
    def toH5(self, file: h5py.File, group_name: str) -> h5py.Group:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter toH5")

    @classmethod
    def Linear(cls, inputDim: int, outputDim: int, activationFunction: Callable = lambda e: e):
        return NeuralLayer.Linear(inputDim, outputDim, activationFunction)

    @classmethod
    def Conv2d(cls, C_in: int, C_out: int, kH: int, kW: int, activationFunction: Callable = lambda e: e, stride = 1):
        return ConvolutionalLayer.random(C_in, C_out, kH, kW, activationFunction, stride)

    @classmethod 
    def MaxPooling(cls, shape: tuple):
        return PoolingLayer(shape, "max")
    
    @classmethod
    def Flatten(cls):
        return FlattenLayer()

# MARK: NeuralLayer
class NeuralLayer(Layer):
    weights: mx.array
    grad_weights: mx.array
    grad_biais: mx.array
    biais: mx.array
    func: Callable
    funcPrime: Callable
    optimizer: str
    
    def __init__(self, weights_: mx.array, biais_: mx.array, activationFunction: Callable, optimizer: str = "adam") -> None:
        super().__init__()
        self.func = activationFunction
        self.funcPrime = prime(self.func)
        self.weights = weights_
        self.biais = biais_
        self.optimizer = optimizer.lower()

        if self.optimizer == "adam":
            self._t        = 0
            self._m_w      = None   # 1er moment weights
            self._v_w      = None   # 2ème moment weights
            self._m_b      = None
            self._v_b      = None
    
    @classmethod
    def Linear(cls, inputDim: int, outputDim: int, activationFunction: Callable | str = lambda e: e):
        if isinstance(activationFunction, str):
            activationFunction = ACTIVATIONS[activationFunction.lower()]

        limit = (6 / (inputDim + outputDim)) ** 0.5

        return cls(
            mx.random.uniform(-limit, limit, (outputDim, inputDim)),
            mx.zeros((outputDim, 1)),
            activationFunction
        )
    
    @classmethod
    def fromH5(cls, grp: h5py.Group) -> "NeuralLayer":
        w = mx.array(grp["weights"][:]) # type: ignore
        b = mx.array(grp["biais"][:]) # type: ignore
        f: str = grp.attrs["func"] # type: ignore

        return cls(w, b, ACTIVATIONS[f.lower()])
    
    def copy(self):
        return NeuralLayer(mx.array(self.weights), mx.array(self.biais), self.func)
    
    def copy_from(self, other: 'NeuralLayer'): # type: ignore
        self.weights = mx.array(other.weights)
        self.biais   = mx.array(other.biais)

    def __str__(self) -> str:
        return f"NeuralLayer<{self.dim} -- {self.weights.shape} --- {self.biais.shape} -- {self.func}>"

    def __call__(self, X: mx.array) -> mx.array:
        Z = X @ mx.transpose(self.weights) + mx.transpose(self.biais)
        A = self.func(Z)
        if self.training:
            self.last_X = X
            self.last_Z = Z
        
        return A

    def backward(self, delta: mx.array, apply_activation_prime: bool = True) -> mx.array:
        if apply_activation_prime:
            delta_z = self.funcPrime(self.last_Z) * delta
        else:
            delta_z = delta

        N = self.last_X.shape[0]

        self.grad_weights = mx.transpose(delta_z) @ self.last_X / N
        self.grad_biais = mx.mean(delta_z, axis=0, keepdims=True).T

        return delta_z @ self.weights
    
    def updateAdam(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1

        # Initialisation paresseuse
        if self._m_w is None:
            self._m_w = mx.zeros_like(self.grad_weights)
            self._v_w = mx.zeros_like(self.grad_weights)
            self._m_b = mx.zeros_like(self.grad_biais)
            self._v_b = mx.zeros_like(self.grad_biais)

        # Mise à jour des moments
        self._m_w = beta1 * self._m_w + (1 - beta1) * self.grad_weights
        self._v_w = beta2 * self._v_w + (1 - beta2) * self.grad_weights ** 2
        self._m_b = beta1 * self._m_b + (1 - beta1) * self.grad_biais
        self._v_b = beta2 * self._v_b + (1 - beta2) * self.grad_biais ** 2

        # Correction du biais (indispensable au début)
        t = self._t
        m_w_hat = self._m_w / (1 - beta1 ** t)
        v_w_hat = self._v_w / (1 - beta2 ** t)
        m_b_hat = self._m_b / (1 - beta1 ** t)
        v_b_hat = self._v_b / (1 - beta2 ** t)

        self.weights = self.weights - lr * m_w_hat / (mx.sqrt(v_w_hat) + eps)
        self.biais   = self.biais   - lr * m_b_hat / (mx.sqrt(v_b_hat) + eps)

    def update(self, learningRate: float):
        if self.optimizer == "adam":
            self.updateAdam(learningRate)
        else:
            self.weights = self.weights - learningRate * self.grad_weights
            self.biais = self.biais - learningRate * self.grad_biais
        
    
    @property
    def dim(self):
        return self.weights.shape

    def getNbParameters(self) -> int:
        return self.weights.shape[0] * self.weights.shape[1] + self.biais.shape[0]
    
    def toH5(self, file: h5py.File, group_name: str) -> h5py.Group:
        grp = file.create_group(group_name)
        grp.attrs["type"] = "NeuralLayer"
        grp.attrs["func"] = self.func.__name__
        grp.create_dataset("weights", data=to_numpy(self.weights))
        grp.create_dataset("biais",   data=to_numpy(self.biais))

        return grp

# MARK: ConvolutionalLayer
class ConvolutionalLayer(Layer):
    kernel: mx.array
    func: Callable
    prime: Callable
    stride: int
    optimizer: str

    def __init__(self, kernel_: mx.array, func_: Callable = lambda e: e, stride = 1, optimizer: str = "adam"):
        super().__init__()
        self.kernel = kernel_
        self.func = func_
        self.funcPrime = prime(func_)
        self.stride = stride
        self.optimizer = optimizer.lower()

        if self.optimizer == "adam":
            self._t        = 0
            self._m      = None
            self._v      = None
        
    @classmethod
    def random(cls, C_in: int, C_out: int, kH: int, kW: int, activationFunction: Callable = lambda e: e, stride: int = 1):
        scale = (2 / (kH * kW * C_in)) ** 0.5
        return cls(
            mx.random.normal((C_out, kH, kW, C_in)) * scale,
            activationFunction, stride
        )
    
    @classmethod
    def fromH5(cls, grp: h5py.Group) -> "ConvolutionalLayer":
        k = mx.array(grp["kernel"][:]) # type: ignore
        f: str = grp.attrs["func"] # type: ignore
        stride: int = grp.attrs["stride"] # type: ignore

        return cls(k, ACTIVATIONS[f.lower()], stride)
    
    def copy(self):
        return ConvolutionalLayer(mx.array(self.kernel), self.func, self.stride)

    def copy_from(self, other: 'ConvolutionalLayer'): # type: ignore
        self.kernel = mx.array(other.kernel)

    def __str__(self) -> str:
        return f"ConvolutionalLater <{self.kernel.shape}>"


    def __call__(self, X: mx.array) -> mx.array:
        if len(X.shape) == 3:
            X = mx.array([X])
        Z = mx.conv2d(X, self.kernel, stride=self.stride)
        # print("Dense Z min/max:", mx.min(Z).item(), mx.max(Z).item())
        if self.training:
            self.last_X = X
            self.last_Z = Z
        return self.func(Z)
    

    def backward(self, delta: mx.array) -> mx.array:
        delta_z = delta * self.funcPrime(self.last_Z)
        Cout, kH, kW, Cin = self.kernel.shape
        B = self.last_X.shape[0]

        if self.stride == 1:
            # ── Astuce batch↔canal : zéro boucle Python, un seul mx.conv2d ──
            X_t = mx.transpose(self.last_X, (3, 1, 2, 0))   # (Cin,  H,    W,    B)
            d_t = mx.transpose(delta_z,     (3, 1, 2, 0))   # (Cout, Hout, Wout, B)
            g   = mx.conv2d(X_t, d_t)                        # (Cin,  kH,   kW,   Cout)
            self.grad = mx.transpose(g, (3, 1, 2, 0)) / B   # (Cout, kH,   kW,   Cin)
        else:
            # ── im2col vectorisé (kH×kW boucles, pas H_out×W_out) ────────────
            X_col     = mx._im2col_strided(self.last_X, kH, kW, self.stride)  # (B, Ho*Wo, kH*kW*Cin)
            delta_col = delta_z.reshape(B, -1, Cout)                       # (B, Ho*Wo, Cout)
            grad      = mx.matmul(mx.transpose(X_col, (0, 2, 1)), delta_col)  # (B, kH*kW*Cin, Cout)
            grad      = mx.mean(grad, axis=0)
            self.grad = mx.transpose(grad, (1, 0)).reshape(Cout, kH, kW, Cin)

        kernel_t   = mx.transpose(self.kernel, (3, 1, 2, 0))
        grad_input = mx.conv_transpose2d(delta_z, kernel_t, stride=self.stride)
        return match_spatial_shape(grad_input, self.last_X.shape)
    
    def updateAdam(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        if self._m is None:
            self._m = mx.zeros_like(self.grad)
            self._v = mx.zeros_like(self.grad)

        self._m = beta1 * self._m + (1 - beta1) * self.grad
        self._v = beta2 * self._v + (1 - beta2) * self.grad ** 2

        m_hat = self._m / (1 - beta1 ** self._t)
        v_hat = self._v / (1 - beta2 ** self._t)

        self.kernel = self.kernel - lr * m_hat / (mx.sqrt(v_hat) + eps)

    def update(self, learningRate: float):
        if self.optimizer == "adam":
            self.updateAdam(learningRate)
        else:
            self.kernel = self.kernel - learningRate * self.grad
    
    def getNbParameters(self) -> int:
        a, b, c, d = self.kernel.shape
        return a * b * c * d
    
    def toH5(self, file: h5py.File, group_name: str) -> h5py.Group:
        grp = file.create_group(group_name)
        grp.attrs["type"] = "ConvolutionalLayer"
        grp.attrs["func"] = self.func.__name__
        grp.attrs["stride"] = self.stride
        grp.create_dataset("kernel", data=to_numpy(self.kernel))

        return grp
    

# MARK: PoolingLayer
class PoolingLayer(Layer):
    shape: tuple
    typ: str
    
    def __init__(self, shape_: tuple, typ_: str):
        super().__init__()
        self.shape = shape_
        self.typ = typ_
    
    @classmethod
    def fromH5(cls, grp: h5py.Group) -> "PoolingLayer":
        typ = grp.attrs["typ"] # type: ignore
        shape = grp.attrs["shape"] # type: ignore

        return cls(shape, typ) # type: ignore
    
    def copy(self):
        return PoolingLayer(self.shape, self.typ)
    
    def copy_from(self, other):
        pass  # pas de poids à copier

    def __str__(self) -> str:
        return f"PoolingLayer <{self.typ} -- {self.shape}>"

    def max2d(self, X: mx.array):
        n, m = X.shape
        nn, nm = n // self.shape[0], m // self.shape[1]
        X_np = to_numpy(X.tolist()) 
        r = np.zeros((nn, nm))
        grad = np.zeros((n, m))
        
        for i in range(nn):
            for j in range(nm):
                bi, bj = self.shape[0] * i, self.shape[1] * j
                patch = X_np[bi:bi+self.shape[0], bj:bj+self.shape[1]]
                max_val = np.max(patch)
                max_pos = np.unravel_index(np.argmax(patch), patch.shape)
                r[i, j] = max_val
                grad[bi + max_pos[0], bj + max_pos[1]] = 1.0
        
        return mx.array(r), grad
    
    def max(self, X: mx.array):
        B, H, W, C = X.shape
        pH, pW = self.shape

        H_out = H // pH
        W_out = W // pW

        # On ignore les bords non divisibles, comme ton code actuel
        X_crop = X[:, :H_out * pH, :W_out * pW, :]

        # (B, H_out, pH, W_out, pW, C)
        X_blocks = X_crop.reshape(B, H_out, pH, W_out, pW, C)

        # On veut max sur pH et pW
        out = mx.max(X_blocks, axis=(2, 4))

        if self.training:
            # masque des maxima
            out_expanded: mx.array = out[:, :, None, :, None, :]
            mask_blocks: mx.array = (X_blocks == out_expanded) # type: ignore

            # Si égalités, on garde potentiellement plusieurs max.
            # Pour MNIST ça ne gêne pas trop, mais voir remarque plus bas.
            self.grad = mask_blocks.reshape(B, H_out * pH, W_out * pW, C)

            # On garde la shape d'entrée pour le backward
            self.input_shape = X.shape

        return out

    def __call__(self, X: mx.array) -> mx.array:
        if self.typ == "max":
            return self.max(X)
        raise ValueError(f"Type de pooling inconnu : {self.typ}")
    
    def backward(self, delta: mx.array) -> mx.array:
        if self.typ != "max":
            raise ValueError(f"Type de pooling inconnu : {self.typ}")

        pH, pW = self.shape
        B, H_in, W_in, C = self.input_shape

        H_out = delta.shape[1]
        W_out = delta.shape[2]

        delta_blocks = delta[:, :, None, :, None, :]
        delta_up = mx.broadcast_to(delta_blocks, (B, H_out, pH, W_out, pW, C))
        delta_up = delta_up.reshape(B, H_out * pH, W_out * pW, C)

        grad_crop = delta_up * self.grad

        bottom_pad = H_in - grad_crop.shape[1]
        right_pad = W_in - grad_crop.shape[2]

        if bottom_pad > 0 or right_pad > 0:
            grad_crop = mx.pad(
                grad_crop,
                [(0, 0), (0, bottom_pad), (0, right_pad), (0, 0)]
            )

        return grad_crop

    def getNbParameters(self) -> int:
        return 0
    
    def toH5(self, file: h5py.File, group_name: str) -> h5py.Group:
        grp = file.create_group(group_name)
        grp.attrs["type"] = "PoolingLayer"
        grp.attrs["typ"] = self.typ
        grp.attrs["shape"] = self.shape

        return grp
    
# MARK: FLattenLayer
class FlattenLayer(Layer):
    last_dim: tuple
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def fromH5(cls, grp: h5py.Group) -> "FlattenLayer":
        return cls()
    
    def copy(self):
        return FlattenLayer()
    
    def copy_from(self, other):
        pass  # pas de poids à copier

    def __str__(self) -> str:
        return "FlattenLayer<>"

    def __call__(self, X: mx.array):
        Z = X.reshape(X.shape[0], -1)

        if self.training:
            self.last_dim = X.shape

        return Z

    def backward(self, delta):
        return delta.reshape(self.last_dim)
    
    def getNbParameters(self) -> int:
        return 0
    
    def toH5(self, file: h5py.File, group_name: str) -> h5py.Group:
        grp = file.create_group(group_name)
        grp.attrs["type"] = "FlattenLayer"

        return grp