from . import mx
from .other import *
import numpy as np
from collections.abc import Callable
from abc import ABC, abstractmethod
import h5py
import sys


def _to_numpy(x):
    """Convertit un array (CuPy, MLX, NumPy) vers np.ndarray CPU."""
    if hasattr(x, 'get'): 
        return x.get()
    if hasattr(x, 'tolist'):
        try:
            return np.array(x.tolist())
        except Exception:
            pass
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
class Layer(ABC):
    weights: mx.array
    grad: mx.array
    training: bool
    last_X: mx.array
    last_Z: mx.array

    registry = {}
    _h5_ignore = {
        "grad",
        "last_X",
        "last_Z",
        "training"
    }

    def __init_subclass__(cls):
        super().__init_subclass__()
        Layer.registry[cls.__name__] = cls

    def _init_layer_state(self):
        self.training = True
        self.grad = None # type: ignore
        self.last_X = None # type: ignore
        self.last_Z = None # type: ignore

    def __init__(self) -> None:
        self._init_layer_state()


    def toH5File(self, file: h5py.File, group_name: str) -> h5py.Group:
        grp = file.create_group(group_name)
        self.toH5(grp)

        return grp

    def toH5(self, grp: h5py.Group):
        grp.attrs["type"] = type(self).__name__

        for key, value in self.__dict__.items():
            if key in self._h5_ignore:
                continue

            if isinstance(value, list) and value and isinstance(value[0], Layer):
                subgrp = grp.create_group(key)
                for i, layer in enumerate(value):
                    layer.toH5(subgrp.create_group(str(i)))
                continue

            if isinstance(value, mx.array):
                grp.create_dataset(key, data=np.array(value))
            elif isinstance(value, (int, float, str, bool)):
                grp.attrs[key] = value

    @classmethod
    def fromH5(cls, grp) -> "Layer":
        typ = grp.attrs["type"]
        cls = Layer.registry[typ]

        obj = cls.__new__(cls)
        obj._init_layer_state()

        # attrs simples
        for key, value in grp.attrs.items():
            if key != "type":
                setattr(obj, key, value)

        # datasets + groupes
        for key in grp.keys():
            item = grp[key]

            if isinstance(item, h5py.Group):
                value = [
                    Layer.fromH5(item[k])
                    for k in sorted(item.keys(), key=int)
                ]
            else:
                value = mx.array(item[()])

            setattr(obj, key, value)

        return obj
        
    
    def copy(self) -> 'Layer':
        raise NotImplementedError(f"{type(self).__name__} doit implémenter copy")
    
    def copy_from(self, other: 'Layer'):
        if type(self) != type(other):
            raise ValueError("Il n'est pas possible de copier depuis une classe différente")
        for key, value in other.__dict__.items():
            if key in self._h5_ignore or key.startswith("last"):
                continue

            setattr(self, key, value)


    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.__dict__}>"

    def __repr__(self) -> str:
        return str(self)

    def __call__(self, X: mx.array) -> mx.array:
        return self.forward(X)    

    @abstractmethod
    def forward(self, X: mx.array) -> mx.array:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter forward")

    def backward(self, delta: mx.array) -> mx.array:
        raise NotImplementedError(f"{type(self).__name__} doit implémenter backward")


    def modules(self):
        yield self

        for key, value in self.__dict__.items():
            if key in self._h5_ignore:
                continue

            if isinstance(value, Layer):
                yield value
                yield from value.modules()
            elif isinstance(value, list):
                for x in value:
                    if isinstance(x, Layer):
                        yield x
                        yield from x.modules()

    def train(self):
        self.training = True
        for m in self.modules():
            m.training = True

    def eval(self):
        mx.grad
        self.training = False
        for m in self.modules():
            m.training = False

    def update(self, learningRate: float, optimizer: str = "sgd"):
        if self.weights is not None and self.grad is not None:
            self.weights = self.weights - learningRate * self.grad

    def getOutputShape(self, inputShape):
        return self(mx.zeros(inputShape)).shape
    
    def getNbParameters(self):
        total = 0

        for key, value in self.__dict__.items():
            if key in self._h5_ignore:
                continue

            if isinstance(value, mx.array):
                total += value.size

            elif isinstance(value, Layer):
                total += value.getNbParameters()

            elif isinstance(value, list):
                for x in value:
                    if isinstance(x, Layer):
                        total += x.getNbParameters()

        return total
    
    @classmethod
    def Sequential(cls, layers: list["Layer"]):
        return SequentialLayer(layers)

    @classmethod
    def Linear(cls, inputDim: int, outputDim: int, activationFunction: Callable = lambda e: e, useBiais: bool = True):
        return NeuralLayer.Linear(inputDim, outputDim, activationFunction, useBiais)
    
    @classmethod
    def ReLU(cls):
        return FuncLayer(ReLU)
    
    @classmethod
    def Softmax(cls):
        return FuncLayer(softmax)
    
    @classmethod
    def Sigmoid(cls):
        return FuncLayer(sigmoid)
    
    @classmethod
    def GeLU(cls):
        return FuncLayer(GeLU)
    
    @classmethod
    def Embedding(cls, vocab_size: int, output_dim: int):
        return EmbeddingLayer.Embedding(vocab_size, output_dim)
    
    @classmethod
    def Dropout(cls, p: float = 0.5):
        return DropoutLayer(p)
    
    @classmethod
    def Normalization(cls):
        return NormalizationLayer()
    
    @classmethod
    def Residual(cls, layer: "Layer"):
        return ResidualLayer(layer)

    @classmethod
    def Conv2d(cls, C_in: int, C_out: int, kH: int, kW: int, activationFunction: Callable = lambda e: e, stride = 1):
        return ConvolutionalLayer.Conv(C_in, C_out, kH, kW, activationFunction, stride)

    @classmethod 
    def MaxPooling(cls, shape: tuple):
        return PoolingLayer(shape, "max")
    
    @classmethod
    def Flatten(cls):
        return FlattenLayer()


# MARK: SequentialLayer
class SequentialLayer(Layer):
    layers: list[Layer]

    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = layers
    
    def forward(self, X: mx.array) -> mx.array:
        for layer in self.layers:
            X = layer(X)
        
        return X
    
    def backward(self, delta: mx.array) -> mx.array:
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        
        return delta
    
    def update(self, learningRate: float, optimizer: str = "adam"):
        for layer in self.layers:
            layer.update(learningRate, optimizer)

    def getNbParameters(self) -> int:
        return sum(layer.getNbParameters() for layer in self.layers)


# MARK: NeuralLayer
class NeuralLayer(Layer):
    weights: mx.array
    biais: mx.array
    _weightsT: mx.array
    _biaisT: mx.array

    func: Callable
    funcPrime: Callable
    funcName: str
    optimizer: str

    grad_weights: mx.array
    grad_biais: mx.array
    

    _h5_ignore = Layer._h5_ignore | {"weightsT", "biaisT", "_weigthsT", "_biaisT", "optimizer", "grad_weights", "grad_biais", "_t", "_m_w", "_v_w", "_m_b", "_v_v"}
    
    def __init__(self, weights_: mx.array, biais_: mx.array, activationFunction: Callable, useBiais: bool = True, optimizer: str = "adam") -> None:
        super().__init__()
        self.weights = weights_
        self.biais = biais_
        self._weightsT = mx.transpose(weights_)
        self._biaisT = mx.transpose(biais_)

        self.func = activationFunction
        self.funcPrime = prime(self.func)
        self.funcName = self.func.__name__
        
        self.useBiais = useBiais
        self.optimizer = optimizer.lower()

        if self.optimizer == "adam":
            self._t        = 0
            self._m_w      = None   # 1er moment weights
            self._v_w      = None   # 2ème moment weights
            self._m_b      = None
            self._v_b      = None

    @property
    def weightsT(self):
        if self._weightsT is not None:
            return self._weightsT
        self._weightsT = mx.transpose(self.weights)

        return self._weightsT
    
    @property
    def biaisT(self):
        if self._biaisT is not None:
            return self._biaisT
        self._biaisT = mx.transpose(self.biais)

        return self._biaisT
    
    @classmethod
    def Linear(cls, inputDim: int, outputDim: int, activationFunction: Callable | str = lambda e: e, useBiais: bool = True):
        if isinstance(activationFunction, str):
            activationFunction = ACTIVATIONS[activationFunction.lower()]

        limit = (6 / (inputDim + outputDim)) ** 0.5

        return cls(
            mx.random.uniform(-limit, limit, (outputDim, inputDim)),
            mx.zeros((outputDim, 1)),
            activationFunction,
            useBiais
        )

    
    def copy(self):
        return NeuralLayer(mx.array(self.weights), mx.array(self.biais), self.func)

    def __str__(self) -> str:
        return f"NeuralLayer<{self.dim} -- {self.biais.shape} -- {self.func}>"

    def forward(self, X: mx.array) -> mx.array:
        Z = X @ self.weightsT + self.biaisT
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
        if self.useBiais:
            self.grad_biais = mx.mean(delta_z, axis=0, keepdims=True).T
        else:
            self.grad_biais = mx.zeros_like(self.biais)

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
        self._v_w = beta2 * self._v_w + (1 - beta2) * self.grad_weights ** 2 # type: ignore
        self._m_b = beta1 * self._m_b + (1 - beta1) * self.grad_biais # type: ignore
        self._v_b = beta2 * self._v_b + (1 - beta2) * self.grad_biais ** 2 # type: ignore

        # Correction du biais (indispensable au début)
        t = self._t
        m_w_hat = self._m_w / (1 - beta1 ** t)
        v_w_hat = self._v_w / (1 - beta2 ** t)
        m_b_hat = self._m_b / (1 - beta1 ** t)
        v_b_hat = self._v_b / (1 - beta2 ** t)

        self.weights = self.weights - lr * m_w_hat / (mx.sqrt(v_w_hat) + eps)
        self.biais   = self.biais   - lr * m_b_hat / (mx.sqrt(v_b_hat) + eps)

    def update(self, learningRate: float, optimizer: str = "sgd"):
        if optimizer == "adam":
            self.updateAdam(learningRate)
        else:
            self.weights = self.weights - learningRate * self.grad_weights
            self.biais = self.biais - learningRate * self.grad_biais
        self._weightsT = mx.transpose(self.weights)
        self._biaisT = mx.transpose(self.biais)
        
    
    @property
    def dim(self):
        return self.weights.shape
    
    def getOutputShape(self, inputShape):
        B = 1
        c = 0
        if len(inputShape) == 1:
            c = inputShape[0]
        elif len(inputShape) == 2:
            B, c = inputShape
        else:
            raise ValueError("L'input d'une Neural Layer doit être flatten")
        
        if c != self.dim[1]:
            raise ValueError(f"Ce réseau ne prend que des entrées de dimensions {self.dim[1]}, pas {c}")
        
        return (B, self.dim[0])


    def getNbParameters(self) -> int:
        b = self.biais.shape[0] if self.useBiais else 0
        return self.weights.size + b
    

# MARK: FuncLayer
class FuncLayer(Layer):
    func: Callable
    funcPrime: Callable
    funcName: str

    def __init__(self, func: Callable):
        super().__init__()
        self.func = func
        self.funcPrime = prime(func)
        self.funcName = func.__name__

    def forward(self, X: mx.array):
        Z = self.func(X)
        if self.training:
            self.last_X = X
            self.last_Z = Z
        return Z
    
    def backward(self, delta: mx.array):
        return self.funcPrime(self.last_Z) * delta
    
    def copy(self):
        return FuncLayer(self.func)

    def __str__(self) -> str:
        return f"FuncLayer{self.func}"
    
    def getNbParameters(self) -> int:
        return 0
    
# MARK: EmbeddingLayer
class EmbeddingLayer(Layer):
    weights: mx.array
    grad: mx.array

    def __init__(self, weights: mx.array):
        if sys.platform != "darwin":
            raise NotImplementedError("Cette classe n'est actuellement implémentée que pour foncitonner sur Apple Silicon avec MLX")
        super().__init__()
        self.weights = weights

    @classmethod
    def Embedding(cls, vocab_size: int, output_dim: int):
        limit = (6 / (vocab_size + output_dim)) ** 0.5
        return cls(mx.random.uniform(-limit, limit, (vocab_size, output_dim)))
    
    
    def copy(self):
        return EmbeddingLayer(mx.array(self.weights))

    def forward(self, X: mx.array) -> mx.array:
        Z = self.weights[X]
        if self.training:
            self.last_X = X
            self.last_Z = Z
        return Z
    
    def backward(self, delta: mx.array) -> mx.array:
        dW = mx.zeros(self.weights.shape)
        grad = dW.at[self.last_X].add(delta)
        self.grad = grad

        return mx.zeros_like(delta)
        
    
    @property
    def dim(self):
        return self.weights.shape
    
    def getOutputShape(self, inputShape):
        B = 1
        c = 0
        if len(inputShape) == 1:
            c = inputShape[0]
        elif len(inputShape) == 2:
            B, c = inputShape
        else:
            raise ValueError("L'input d'un Embedding Layer doit être flatten")
        
        if c != self.dim[1]:
            raise ValueError(f"Ce réseau ne prend que des entrées de dimensions {self.dim[1]}, pas {c}")
        
        return (B, self.dim[0])


    def getNbParameters(self) -> int:
        return self.weights.size
    


# MARK: DropoutLayer
class DropoutLayer(Layer):
    p: float
    last_mask: mx.array
    q: float

    _h5_ignore = Layer._h5_ignore | {"last_mask"}

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.p = probability
        self.last_mask = None # type: ignore
        self.q = 1 / (1. - self.p)
    
    def copy(self):
        return DropoutLayer(self.p)
    

    def forward(self, X: mx.array) -> mx.array:
        if not self.training:
            return X
        
        mask = (mx.random.uniform(shape=X.shape) > self.p).astype(X.dtype) * self.q
        Z = mask * X
        self.last_X = X
        self.last_Z = Z
        self.last_mask = mask
        return Z
    
    def backward(self, delta: mx.array) -> mx.array:
        return self.last_mask * delta
    
    def getOutputShape(self, inputShape):
        return inputShape
    



# MARK: NormalizationLayer
class NormalizationLayer(Layer):
    EPS = 1e-8

    def __init__(self):
        super().__init__()

    def forward(self, X):
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)

        self.mean = mean
        self.var = var

        return (X - mean) / mx.sqrt(var + self.EPS)

    def backward(self, delta):
        return delta


# MARK: ResidualLayer
class ResidualLayer(Layer):
    def __init__(self, layer: Layer):
        super().__init__()
        self.layer = layer

    def forward(self, X):
        self.X = X
        return X + self.layer(X)

    def backward(self, delta):
        return delta + self.layer.backward(delta)

    def update(self, learningRate: float, optimizer: str = "adam"):
        self.layer.update(learningRate, optimizer)

    def getNbParameters(self) -> int:
        return self.layer.getNbParameters()

# MARK: ConvolutionalLayer
class ConvolutionalLayer(Layer):
    kernel: mx.array
    func: Callable
    prime: Callable
    stride: int
    optimizer: str

    _h5_ignore = Layer._h5_ignore | {"kernel"}

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
    def Conv(cls, C_in: int, C_out: int, kH: int, kW: int, activationFunction: Callable = lambda e: e, stride: int = 1):
        scale = (2 / (kH * kW * C_in)) ** 0.5
        return cls(
            mx.random.normal((C_out, kH, kW, C_in)) * scale,
            activationFunction, stride
        )

    def copy(self):
        return ConvolutionalLayer(mx.array(self.kernel), self.func, self.stride)

    def __str__(self) -> str:
        return f"ConvolutionalLater <{self.kernel.shape}>"

    def forward(self, X: mx.array) -> mx.array:
        if len(X.shape) == 3:
            X = mx.array([X])
        Z = mx.conv2d(X, self.kernel, stride=self.stride)
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
            X_col     = im2col_strided(self.last_X, kH, kW, self.stride)  # (B, Ho*Wo, kH*kW*Cin)
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
        self._v = beta2 * self._v + (1 - beta2) * self.grad ** 2 # type: ignore

        m_hat = self._m / (1 - beta1 ** self._t)
        v_hat = self._v / (1 - beta2 ** self._t)

        self.kernel = self.kernel - lr * m_hat / (mx.sqrt(v_hat) + eps)

    def update(self, learningRate: float, optimizer: str = "sgd"):
        if optimizer == "adam":
            self.updateAdam(learningRate)
        else:
            self.kernel = self.kernel - learningRate * self.grad

    def getOutputShape(self, inputShape):
        B = 1
        c_in = 1
        if len(inputShape) == 2:
            h, w = inputShape
        elif len(inputShape) == 3:
            h, w, c_in = inputShape
        else:
            B, h, w, c_in = inputShape

        if c_in != self.kernel.shape[3]:
            raise ValueError(f"L'entrée de cette convolution doit avoir {self.kernel.shape[3]} canaux, pas {c_in}")

        c_out, kh, kw, _ = self.kernel.shape

        h_out = (h - kh) // self.stride + 1
        w_out = (w - kw) // self.stride + 1

        return (B, h_out, w_out, c_out)
    
    def getNbParameters(self) -> int:
        a, b, c, d = self.kernel.shape
        return a * b * c * d
    
    

# MARK: PoolingLayer
class PoolingLayer(Layer):
    shape: tuple
    typ: str
    
    def __init__(self, shape_: tuple, typ_: str):
        super().__init__()
        self.shape = shape_
        self.typ = typ_
    
    
    def copy(self):
        return PoolingLayer(self.shape, self.typ)
    

    def __str__(self) -> str:
        return f"PoolingLayer <{self.typ} -- {self.shape}>"

    def max2d(self, X: mx.array):
        n, m = X.shape
        nn, nm = n // self.shape[0], m // self.shape[1]
        X_np = _to_numpy(X)         
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

    def forward(self, X: mx.array) -> mx.array:
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
    
    def getOutputShape(self, inputShape):
        B, h, w, c_in = inputShape
        
        return (B, h // self.shape[0], w // self.shape[1], c_in)

    def getNbParameters(self) -> int:
        return 0

    
# MARK: FLattenLayer
class FlattenLayer(Layer):
    last_dim: tuple
    
    _h5_ignore = Layer._h5_ignore | {"last_dim"}

    def __init__(self):
        super().__init__()
    
    
    def copy(self):
        return FlattenLayer()

    def __str__(self) -> str:
        return "FlattenLayer<>"

    def forward(self, X: mx.array):
        Z = X.reshape(X.shape[0], -1)

        if self.training:
            self.last_dim = X.shape

        return Z

    def backward(self, delta):
        return delta.reshape(self.last_dim)
    
    def getOutputShape(self, inputShape):
        B, h, w, c_in = inputShape
        
        return (B, h*w*c_in)
    