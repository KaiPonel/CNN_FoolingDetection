import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib import image
import gc

from memory_profiler import profile

def save_model_weights(model, model_name="cnn", dataset="mnist", base_path="/project/FoolingDetection/models"):
    """
        Saves the model weights to the disk \n
        `model`: untrained tf.model \n
        `model_name`: name of the model, e.g. "cnn", "mlp" \n
        `dataset`: dataset the model was trained on e.g "mnist", "cifar10" \n
        `base_path`: base directory where the weights should be stored \n
    """
    if model_name not in ["cnn", "mlp", "imagenet"]:
        raise Exception("Invalid value for `model_type`. Got {}, expected [`cnn`, `mlp`, `imagenet`]".format(model_type))
    if dataset not in ["mnist", "cifar10", "imagenet"]:
        raise Exception("Invalid value for `dataset`. Got {}, expected [`mnist`, `cifar10`, `imagenet`]".format(dataset))

    weights_dir = os.path.join(base_path, model_name, dataset)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    weights_path = os.path.join(weights_dir, 'weights.h5')
    model.save_weights(weights_path)


def load_model_weights(model, model_name, dataset, base_path="/project/FoolingDetection/models"):
    """
        Loads the weights of a model. \n
        `model`: untrained tf.model \n
        `model_name`: name of the model, e.g. "cnn", "mlp" \n
        `dataset`: dataset the model was trained on e.g "mnist", "cifar10" \n
        `base_path`: base directory of weights directory \n
    """
    weights_dir = os.path.join(base_path, model_name, dataset)
    weights_path = os.path.join(weights_dir, 'weights.h5')
    if not os.path.exists(weights_dir):
        raise Exception("Invalid path: {}; Please check the parameters `model_name`, `dataset` and `base_path`".format(weights_dir))
    model.load_weights(weights_path)
    return model

"""" Memory Introspection stuff """
#from memory_profiler import profile
#@profile
def save_grids(grids_list, dataset, model, attack="correct", override=False, base_dir="/project/FoolingDetection/grids"):
    """
        Save grids to disk following the path `base_dir/dataset/model/attack/technique/index` \n
        `grids_list`: array of grids to be stored \n
        `base_dir`: base directory where grids should be stored \n
        `dataset`: dataset the model was trained / explanined on \n
        `model`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
        `attack`: type of attack performed; "correct" means no attack (Default) \n
        `override`: Determines whether previous should be overwritten or not.
    """
    # Define save directory
    base_dir = os.path.join(base_dir, dataset, model, attack)
    
    if os.path.exists(base_dir) and not override:
        raise Exception("The directory already exists and `override=False` was passed. Please make sure that you want to do this, this will delete any previous results.")
    
    # Iterate over all grids and save them to corresponding subfolder
    for grids, subfolder in grids_list:
        subfolder_path = os.path.join(base_dir, subfolder)

        # Create subdirectory if it does not exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Save each grid as an image
        for i, grid in enumerate(grids):
            plt.imshow(grid, cmap='hot', interpolation='nearest')
            plt.savefig(os.path.join(subfolder_path, f"{i}.png"))
            plt.clf()
            plt.close()
            del grid
            gc.collect()
        del grids
        gc.collect()


def load_grids(base_dir, dataset, model, attack, explanation_technique, max_images=300):
    """
         Load grids from disk following the path `base_dir/dataset/model/attack/technique/` \n
        `base_dir`: base directory where grids should be stored \n
        `dataset`: dataset the model was trained / explanined on \n
        `model`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
        `attack`: type of attack performed; "correct" means no attack (Default) \n
        `explanation_technique`: explanation technique for which the grids should be loaded \n
    """
    # Define load directory
    load_dir = os.path.join(base_dir, dataset, model, attack, explanation_technique)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from images
    grids = []
    for i, file in enumerate(sorted(os.listdir(load_dir), key=lambda x: int(os.path.splitext(x)[0]))):
        if i >= max_images:
            break
        grid = image.imread(os.path.join(load_dir, file))
        grids.append(grid)
    return grids

def load_all_grids(base_dir, dataset, model, attack, max_images=300):
    """
    Load grids from all explanation techniques of a specified directory.
    
    `base_dir`: base directory where grids should be stored \n
    `dataset`: dataset the model was trained/explained on \n
    `model`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
    `attack`: type of attack performed; "correct" means no attack \n
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the list of grids in that directory.
    """
    # Define load directory
    load_dir = os.path.join(base_dir, dataset, model, attack)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from all techniques
    all_grids = {}
    for techniques in os.listdir(load_dir):
        techniques_path = os.path.join(load_dir, techniques)
        if os.path.isdir(techniques_path):
            grids = load_grids(base_dir, dataset, model, attack, techniques, max_images=max_images)
            all_grids[techniques] = grids
    
    return all_grids

def load_all_grids_randparams(base_dir, dataset, model, attack, max_images=300):
    """
    Load grids from all explanation techniques and for the RandomParams attack
    
    `base_dir`: base directory where grids should be stored \n
    `dataset`: dataset the model was trained/explained on \n
    `model`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
    `attack`: type of attack performed; "correct" means no attack \n
    
    :return: dict
        Dictionary with keys being the name of the attack variants and values being dictionaries with keys being the name of the techniques and values being the list of grids in that directory.
    """
    # Define load directory
    load_dir = os.path.join(base_dir, dataset, model, attack)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from all attack variants and techniques
    all_grids = {}
    # Sort the layers, start with the highest (top-level) layers
    variants = sorted(os.listdir(load_dir), key=lambda x: int(x.split('_')[0]), reverse=True)
    for variant in variants:
        variant_path = os.path.join(load_dir, variant)
        if os.path.isdir(variant_path):
            variant_grids = {}
            for technique in os.listdir(variant_path):
                technique_path = os.path.join(variant_path, technique)
                if os.path.isdir(technique_path):
                    grids = load_grids(base_dir, dataset, model, attack, os.path.join(variant, technique), max_images=max_images)
                    variant_grids[technique] = grids
            all_grids[variant] = variant_grids
    
    return all_grids