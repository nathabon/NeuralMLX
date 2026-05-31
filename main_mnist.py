from neuralNetwork2 import *
import h5py


dataset_filename = "datasets/mnist/1_(28, 28)_enhanced.h5"
file = h5py.File(dataset_filename, "r")
dsetX = file["train_X"][:] # type: ignore
dsetY = file["train_Y"][:] # type: ignore

train_X = mx.array(dsetX) / 255.0 # type: ignore
train_Y = mx.array(dsetY) # type: ignore

print(train_X.shape)


# network = NeuralNetwork([
#     Layer.Linear(28*28, 200, sigmoid),
#     Layer.Linear(200, 100, sigmoid),
#     Layer.Linear(100, 10, softmax)
# ])


# network = NeuralNetwork.fromFileH5("saves/mnist/train_full_conv.h5")


network = NeuralNetwork([
    Layer.Conv2d(1, 4, 3, 3, ReLU),
    Layer.MaxPooling((2, 2)),
    Layer.Conv2d(4, 8, 3, 3, ReLU),
    Layer.MaxPooling((2, 2)),
    Layer.Flatten(),
    Layer.Linear(200, 10, softmax)
])

network = NeuralNetwork([
    Layer.Conv2d(1, 8, 5, 5, ReLU, stride=2),
    Layer.Conv2d(8, 16, 3, 3, ReLU, stride=2),
    Layer.Flatten(),
    Layer.Linear(400, 64, ReLU),
    Layer.Linear(64, 10, softmax)
])



save = True
print("Train")
try:
    network.train(train_X, train_Y, nb_epochs=200, learningRate=0.1, batch_size=128)
except KeyboardInterrupt:
    print()
    save = True if input("Voulez vous sauvegarder les paramètres (O, n) ? ") == "o" else False


if save:
    print("Saving data...", end="\r")
    network.saveH5("saves/mnist/train_full_conv_h_2.h5")
    print("All datas have been saved")

