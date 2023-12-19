# This file is used to serve as a data loader for the project.
# This reduces the code in other files

from torch.utils.data import DataLoader, TensorDataset
import torch
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

    return rearrange_to_channels_first(x_train.astype(np.float32)), y_train, rearrange_to_channels_first(x_test.astype(np.float32)), y_test


def load_CIFAR_batch(filename):
    """Load a single batch of the CIFAR-10 dataset"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float").astype(np.float32)
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
    return rearrange_to_channels_first(Xtr), Ytr, rearrange_to_channels_first(Xte), Yte

def load_dataset(name, as_dict=False):
    """
        Loads the dataset specified by `name` \n
        `name`: name of the dataset to load \n
        `as_dict`: if True, returns the dataset as a dictionary, otherwise as a tuple \n

        Returns the dataset specified by `name`
    """
    if name == "mnist":#
        dataset = load_MNIST()
    elif name == "cifar10":
        dataset =  load_CIFAR10()
    else:
        raise Exception(f"Dataset {name} is not supported.")
    
    # If as_dict is True, return the dataset as a dictionary
    if as_dict:
        return {"x_train": dataset[0], "y_train": dataset[1], "x_test": dataset[2], "y_test": dataset[3]}
    # Otherwise, return the dataset as a tuple
    else:
        return dataset


def convert_to_td_and_dl(x, y, batch_size):
    """
        Converts numpy arrays x and y into a TensorDataset and DataLoader 
        This is used for training using pytorch.    
    """
    x, y = map(torch.tensor, (x,y))
    return DataLoader(TensorDataset(x,y), batch_size=batch_size)

def rearrange_to_channels_first(x):
    """
        Rearranges the numpy array x to have the channels first
        This is used for training using pytorch.    
    """
    return x.transpose(0, 3, 1, 2)

