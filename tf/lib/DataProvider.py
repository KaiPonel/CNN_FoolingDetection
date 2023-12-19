# This file is used to serve as a data loader for the project.
# This reduces the code in other files


import numpy as np
import pickle
import os

def trim_lists(x, y, amount):
    """
    Trims two lists x and y to contain only the first `amount` elements
    
    `x`: first list to trim \n
    `y`: second list to trim \n
    `amount`: number of elements to keep in the trimmed lists \n
    
    :return: tuple
        Tuple containing the trimmed versions of x and y
    """
    # Trim x and y to contain only the first `amount` elements
    x_trimmed = x[:amount]
    y_trimmed = y[:amount]
    
    return x_trimmed, y_trimmed





def load_MNIST(dirpath='/data/project/FoolingDetection/mnist.npz'):
    """ 
    Returns: x_train, y_train, x_test, y_test
    """
    with np.load(dirpath) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

    x_train = x_train/255
    x_test = x_test/255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def load_CIFAR_batch(filename):
    """Load a single batch of the CIFAR-10 dataset"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(dirpath="/data/project/FoolingDetection/cifar-10-batches-py"):
    """Load all batches of the CIFAR-10 dataset from a directory"""
    xs = []
    ys = []
    for b in range(1, 5):
        f = os.path.join(dirpath, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(dirpath, 'test_batch'))
    return Xtr, Ytr, Xte, Yte