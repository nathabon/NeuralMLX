import backend_cupy as mx
from other import *
from typing import Callable
import random
import time
from neuralLayers import *
import h5py


def max_indice(iterable: mx.array) -> tuple:
    flat = iterable.flatten()
    ind = int(mx.argmax(flat).item())
    return ind, float(flat[ind].item())


class NeuralNetwork:
    layers: list[Layer]
    L: int
    cost_function: str

    def __init__(self, layers_: list[Layer]) -> None:
        self.layers = layers_
        self.L = int(len(layers_))

    @classmethod
    def fromFile(cls, file_name, f):
        layers = []
        with open(file_name, "r") as file:
            datas = file.read().split(" ")
        
        i = 1
        while i < len(datas):
            n = int(datas[i-1][1:datas[i-1].index(",")])
            nb = int(datas[i-1][datas[i-1].index(",")+1: -1])
            matrice = []
            for ind in range(nb):
                matrice.append([float(e) for e in datas[i + (ind * n):i + ((ind + 1) * n)]])
            mb = [[float(e)] for e in datas[i + (n * nb):i + nb * n + nb]]
            layers.append(NeuralLayer(mx.array(matrice), mx.array(mb), f))
            i += nb * n + nb + 1
        
        return cls(layers)
    
    @classmethod
    def fromFileH5(cls, file_name):
        layers = []
        with h5py.File(file_name, "r") as f:
            for key in sorted(f.keys()):  # layer_0, layer_1...
                grp = f[key]
                layers.append(Layer.fromH5(grp)) # type: ignore
        return cls(layers)

    @classmethod
    def random(cls, layers_size: tuple, func):
        layers = []
        for i in range(len(layers_size) - 1):
            l = Layer.Linear(layers_size[i], layers_size[i+1], func)
            layers.append(l)
        return cls(layers)

    @classmethod
    def given(cls, weights, biais, func):
        layers = []
        for w, b in zip(weights, biais):
            layers.append(NeuralLayer(mx.array(w), mx.array(b), func))
        return cls(layers)
    
    def copy(self):
        layers = []
        for layer in self.layers:
            layers.append(layer.copy())
        
        return NeuralNetwork(layers)

    # -------------
    # --- Maths ---
    # -------------
    def loss(self, T: mx.array, Y: mx.array) -> mx.array:
        return T - Y

    def cost(self, Y: mx.array, T: mx.array) -> float:
        eps = 1e-8
        Y = mx.clip(Y, eps, 1.0 - eps)
        return float(-mx.mean(mx.sum(T * mx.log(Y), axis=1)).item())

    def costPrime(self, T: mx.array, Y: mx.array) -> mx.array:
        return 2 * self.loss(T, Y)

    def getDelta(self, results_a: mx.array, T: mx.array):
        c_delta: mx.array
        if T is not None:
            T = T.reshape(results_a.shape)

            # Cas softmax + cross-entropy :
            # delta de sortie = prédiction - target
            c_delta = results_a - T
        else:
            c_delta = results_a

        first_backward = True

        for layer in reversed(self.layers):
            if first_backward and isinstance(layer, NeuralLayer) and layer.func == softmax:
                c_delta = layer.backward(c_delta, apply_activation_prime=False)
            else:
                c_delta = layer.backward(c_delta)

            first_backward = False


    def updateWeights(self, learningRate):
        for layer in self.layers:
            layer.update(learningRate)

    def __call__(self, vector: mx.array) -> mx.array:
        last = vector
        for layer in self.layers:
            last = layer(last)
        return last
    
    def eval_mlx(self):
        pass
    
    def freeze(self):
        for layer in self.layers:
            layer.training = False

    
    @no_grad
    def _test_batch(self, test_X: mx.array, test_Y: mx.array):
        nb_good_results = 0

        for x, y in zip(test_X, test_Y):
            res = self(x)
            if max_indice(res)[0] == max_indice(y)[0]:
                nb_good_results += 1
        
        return nb_good_results / len(test_X)
    
    @no_grad
    def test_batch(self, test_X: mx.array, test_T: mx.array, batch_size: int = 256):
        N = test_X.shape[0]
        nb_good_results = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_x = test_X[start:end]  # (B, H, W, C)
            batch_t = test_T[start:end]  # (B, 10)

            batch_y = self(batch_x)

            pred = mx.argmax(batch_y.reshape(end - start, -1), axis=1)
            true = mx.argmax(batch_t.reshape(end - start, -1), axis=1)

            nb_good_results += int(mx.sum(pred == true).item()) # type: ignore
            mx.eval(pred, true)  # libère la mémoire GPU entre batches

        return nb_good_results / N



    def train(self, inputs, desired_outputs, test_X = None, test_Y = None, learningRate: float = 0.25, nb_epochs: int = 10, shuffle: bool = True, batch_size: int = 1):
        data_size = min(len(inputs), len(desired_outputs))
        datas = list(zip(inputs, desired_outputs))

        interval = max(1, data_size // 100)
        for epoch in range(nb_epochs):
            if shuffle:
                random.shuffle(datas)

            ti = time.time()
            nb_good_results = 0
            total_loss = 0.0
            nb_batches = (data_size + batch_size - 1) // batch_size

            for batch_idx in range(nb_batches):
                batch = datas[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                B = len(batch)

                batch_x = mx.stack([x for x, _ in batch], axis=0)
                batch_t = mx.stack([t for _, t in batch], axis=0).reshape(B, -1)

                # Forward 
                result_a = self(batch_x)

                # Précision
                pred_classes = mx.argmax(result_a, axis=1)
                true_classes = mx.argmax(batch_t, axis=1)
                nb_good_results += int(mx.sum(pred_classes == true_classes).item()) # type: ignore

                # Loss
                batch_cost = self.cost(result_a, batch_t)
                total_loss += batch_cost

                # Backward
                self.getDelta(result_a, batch_t)
                self.updateWeights(learningRate)

                # self.eval_mlx()

                it = (batch_idx + 1) * batch_size
                if it % interval == 0:
                    print("[{}{}] {}/{}, precision: {:.3f}%, epoch: {}, avg cost: {:.4f}, time: {:d}m {:d}s".format(
                        '#' * round((batch_idx + 1) / nb_batches * 50),
                        '-' * (50 - round((batch_idx + 1) / nb_batches * 50)),
                        min(it, data_size), data_size,
                        nb_good_results / min(it, data_size) * 100,
                        epoch,
                        total_loss / (batch_idx + 1),
                        int((time.time() - ti) // 60),
                        int((time.time() - ti) % 60)
                    ), end="\r")

            # on compare avec le batch de tests
            if test_X is not None and test_Y is not None:
                precision_test = self.test_batch(test_X, test_Y)
                print("[{}{}] {}/{}, precision: {:.3f}%, epoch: {}, avg cost: {:.4f}, time: {:d}m {:d}s, precisions test: {:.3f}%".format(
                    '#' * 50,
                    '-' * 0,
                    data_size, data_size,
                    nb_good_results / data_size * 100,
                    epoch,
                    total_loss / nb_batches,
                    int((time.time() - ti) // 60),
                    int((time.time() - ti) % 60),
                    precision_test * 100
                ), end="\r")

            print()

    def saveH5(self, file_name="save.h5"):
        with h5py.File(file_name, "w") as f:
            for i, layer in enumerate(self.layers):
                layer.toH5(f, f"layer_{i}")
    
    def getNpParameters(self):
        sum = 0
        for layer in self.layers:
            sum += layer.getNbParameters()
        
        return sum
    
    def get_output_shape(self, input_shape):
        mat = mx.ones(input_shape)
        return self(mat).shape


def get_output_shape(net: NeuralNetwork, input_shape):
    mat = mx.ones(input_shape)
    return net(mat).shape


if __name__ == "__main__":
    # network = NeuralNetwork.given(
    #     [[[0.59778282, 0.74936583], [0.70684852, -0.86490778], [-0.78280734, 0.93244844]],
    #     [[-0.88236882, -0.13081799, -0.06182501]]],
    #     [[[0], [0], [0]], [[0]]],
    #     sigmoid
    # )


    # Vecteurs lignes (n, B) — concaténables sur axis=1
    X = [mx.array([[1., 1.]]).T, mx.array([[0., 1.]]).T, mx.array([[1., 0.]]).T, mx.array([[0., 0.]]).T]
    T = [mx.array([[1.]]),       mx.array([[0.]]),       mx.array([[0.]]),       mx.array([[0.]])]

    # print("Training...")
    # t = time.time()
    # network.train(X, T, sigmoid, nb_epochs=100, learningRate=0.5, shuffle=False, batch_size=1)
    # print(time.time() - t)

    for x, t_val in zip(X, T):
        result = network(x)
        print(f"Input: {x.T.tolist()} → Output: {result[-1].tolist()} (attendu: {t_val.tolist()})")

    network.saveH5("save3.h5")