import os
import h5py
import fidle

import neuralNetwork2 as nn
import mlx.core as mx


def read_dataset(output_dir, dataset_name, scale=1):
    '''
    Reads h5 dataset
    Args:
        filename     : datasets filename
        dataset_name : dataset name, without .h5
    Returns:    
        x_train,y_train, x_test,y_test data, x_meta,y_meta
    '''

    # ---- Read dataset
    #
    chrono=fidle.Chrono()
    chrono.start()
    filename = f'{output_dir}/{dataset_name}.h5'
    with  h5py.File(filename,'r') as f:
        x_train = f['x_train'][:] # type: ignore
        y_train = f['y_train'][:]# type: ignore
        x_test  = f['x_test'][:]# type: ignore
        y_test  = f['y_test'][:]# type: ignore
        x_meta  = f['x_meta'][:]# type: ignore
        y_meta  = f['y_meta'][:]# type: ignore

    # ---- Rescale 
    #
    print('Original shape  :', x_train.shape, y_train.shape)# type: ignore
    x_train,y_train, x_test,y_test = fidle.utils.rescale_dataset(x_train,y_train,x_test,y_test, scale=scale)
    print('Rescaled shape  :', x_train.shape, y_train.shape)

    # ---- Shuffle
    #
    x_train,y_train=fidle.utils.shuffle_np_dataset(x_train,y_train)

    # ---- done
    #
    duration = chrono.get_delay()
    size     = fidle.utils.hsize(os.path.getsize(filename))
    print(f'\nDataset "{dataset_name}" is loaded and shuffled. ({size} in {duration})')
    return x_train,y_train, x_test,y_test, x_meta,y_meta

# ---- Read dataset
output_dir = "datasets/gtsrb/"
dataset_name  = 'set-24x24-RGB-HE-1'


x_train,y_train,x_test,y_test, x_meta,y_meta = read_dataset(output_dir, dataset_name)
train_X, test_X = mx.array(x_train), mx.array(x_test)
train_X = train_X / 255.0
test_X = test_X / 255.0
train_Y = mx.array([nn.zeros_hot(43, x) for x in y_train])
test_Y = mx.array([nn.zeros_hot(43, x) for x in y_test])


# network = nn.NeuralNetwork([
#     nn.Layer.Conv2d(3, 96, 3, 3, nn.ReLU),
#     nn.Layer.MaxPooling((2, 2)),
#     nn.Layer.Conv2d(96, 192, 3, 3, nn.ReLU),
#     nn.Layer.MaxPooling((2, 2)),
#     nn.Layer.Flatten(),
#     nn.Layer.Linear(3072, 1500, nn.ReLU),
#     nn.Layer.Linear(1500, 43, nn.softmax)
# ])

network = nn.NeuralNetwork.fromFileH5(f"saves/gtsrb/save-{dataset_name}-3.h5")

print(network.test_batch(test_X, test_Y))

save = True
print("Train")
try:
    network.train(train_X, train_Y, test_X=test_X, test_Y=test_Y, nb_epochs=100, learningRate=1, batch_size=128)
except KeyboardInterrupt:
    print()
    save = True if input("Voulez vous sauvegarder les paramètres (O, n) ? ") == "o" else False


if save:
    print("Saving data...", end="\r")
    network.saveH5(f"saves/gtsrb/save-{dataset_name}-4.h5")
    print("All datas have been saved")

