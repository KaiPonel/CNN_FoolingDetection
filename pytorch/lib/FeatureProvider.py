import shutil
import torch
import os
import cv2
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA


def save_model_weights(model, model_type="cnn", dataset="mnist", base_path="/project/FoolingDetection/pytorch/models"):
    """
        Saves the model weights to the disk \n
        `model`: trained PyTorch model, stored as state_dict \n
        `model_type`: type of model, e.g. "cnn", "mlp" \n
        `dataset`: dataset the model was trained on e.g "mnist", "cifar10" \n
        `base_path`: base directory where the weights should be stored \n
    """

    # Check if the model type and dataset are valid, otherwise raise an exception (to avoid human error...)
    if model_type not in ["cnn", "mlp", "imagenet"]:
        raise Exception("Invalid value for `model_type`. Got {}, expected [`cnn`, `mlp`, `imagenet`]".format(model_type))
    if dataset not in ["mnist", "cifar10", "imagenet"]:
        raise Exception("Invalid value for `dataset`. Got {}, expected [`mnist`, `cifar10`, `imagenet`]".format(dataset))

    # Define the directory where the weights should be stored, based on the model type and dataset
    weights_dir = os.path.join(base_path, model_type, dataset)

    # Create weight directory if it does not exist
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Save the weights
    weights_path = os.path.join(weights_dir, 'weights.pt')
    torch.save(model, weights_path)


def load_model_weights(model_type, dataset, base_path="/project/FoolingDetection/pytorch/models"):
    """
        Loads the weights of a model. \n
        `model_type`: type of model, e.g. "cnn", "mlp" \n
        `dataset`: dataset the model was trained on e.g "mnist", "cifar10" \n
        `base_path`: base directory of weights directory \n
    """

    # Define the directory where the weights should be stored, based on the model type and dataset
    weights_dir = os.path.join(base_path, model_type, dataset)

    # Check if the directory exists
    if not os.path.exists(weights_dir):
        raise Exception("Invalid path: {}; Please check the parameters `model_name`, `dataset` and `base_path`".format(weights_dir))

    # Load the weights and return them
    weights_path = os.path.join(weights_dir, 'weights.pt')
    return torch.load(weights_path)


def save_grids(grids_dict, dataset, model_type, attack="correct", override=False, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
        Save grids to disk following the path `base_dir/dataset/model/attack/technique/index` \n
        `grids_dict`: dictionary of grids \n
        `base_dir`: base directory where grids should be stored \n
        `dataset`: dataset the model was trained / explanined on \n
        `model_type`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
        `attack`: type of attack performed; "correct" means no attack (Default) \n
        `override`: Determines whether previous should be overwritten or not.
    """

    print("Saving grids to disk...")

    # Define save directory based on the dataset, model type and attack
    base_dir = os.path.join(base_dir, dataset, model_type, attack)
    
    # Check if directory exists and override of previous results is not allowed
    if os.path.exists(base_dir):
        if not override:
            raise Exception("The directory already exists and `override=False` was passed. Please make sure that you want to do this, this will delete any previous results.")
        else:
            print("The directory `{}` already exists and `override=True` was passed. Deleting previous results...".format(base_dir))
            # Delete the directory
            shutil.rmtree(base_dir)

    # Iterate over all grids in the dictionary and save them to corresponding subfolder, couting the index
    
    for subfolder_name, grids in grids_dict.items():
        subfolder_path = os.path.join(base_dir, subfolder_name)

        # Create subdirectory if it does not exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Save each grid as an image
        for i, grid in enumerate(grids):
            grid_np = np.array(grid)
            # If the grids have 3 dimensions, transpose them to the correct format
            if (grid_np.shape[0] == 3):
                grid_np = np.transpose(grid_np, (1, 2, 0))
            #grid_np = np.transpose(grid_np, (1, 2, 0))

            # Every 100 images, print the progress
            if (i+1) % 50 == 0:
                print(f"Saving image {i+1} out of {len(grids)} ({(i+1)/len(grids)*100:.2f}%)")

            # Store image as png on disk without axis, frame - Only the content of the image using cv2
            cv2.imwrite(os.path.join(subfolder_path, f"{i}.png"), grid_np * 255)




def load_grids(dataset, model_type, attack, explanation_technique, max_images=300, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
         Load grids from disk following the path `base_dir/dataset/model/attack/technique/` \n
        `dataset`: dataset the model was trained / explanined on \n
        `model_type`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
        `attack`: type of attack performed; "correct" means no attack (Default) \n
        `explanation_technique`: explanation technique for which the grids should be loaded \n
        `max_images`: maximum number of images to be loaded \n
        `base_dir`: base directory where grids should be stored \n

        :return: list of grids
    """
    # Define load directory based on the provided parameters
    load_dir = os.path.join(base_dir, dataset, model_type, attack, explanation_technique)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from images
    grids = []

    print("Loading grids from disk...")

    for i, file in enumerate(sorted(os.listdir(load_dir), key=lambda x: int(os.path.splitext(x)[0]))):

        # Every 50 images, print the progress
        if (i+1) % 50 == 0:
            print(f"Loading image {i+1} out of {max_images} ({(i+1)/max_images*100:.2f}%)")

        if i >= max_images:
            break
        # Load the image using cv2; if the dataset is mnist, load it as grayscale otherwise load it as RGB
        if dataset == "mnist":
            grid = cv2.imread(os.path.join(load_dir, file), cv2.IMREAD_GRAYSCALE)
        else:
            grid = cv2.imread(os.path.join(load_dir, file))
        grids.append(grid.astype(np.int16))
    return grids


def load_all_grids(dataset, model_type, attack, max_images=300, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
    Load grids from all explanation techniques of a specified directory.
    
    `base_dir`: base directory where grids should be stored \n
    `dataset`: dataset the model was trained/explained on \n
    `model_type`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
    `attack`: type of attack performed; "correct" means no attack \n
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the list of grids in that directory.
    """

    # Define load directory based on the dataset, model type and attack
    load_dir = os.path.join(base_dir, dataset, model_type, attack)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from all techniques
    all_grids = {}
    # "variants" correspond to the different layers of the model which were randomized
    # "techniques" correspond to the different introspection techniques
    for techniques in sorted(os.listdir(load_dir)):
        techniques_path = os.path.join(load_dir, techniques)
        if os.path.isdir(techniques_path):
            grids = load_grids(dataset, model_type, attack, techniques, max_images=max_images, base_dir=base_dir)
            all_grids[techniques] = grids
    
    return all_grids

def load_all_grids_randparams(dataset, model_type, attack, max_images=300, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
    Load grids from all explanation techniques and for the RandomParams attack
    
    `base_dir`: base directory where grids should be stored \n
    `dataset`: dataset the model was trained/explained on \n
    `model_type`: type of model that was used (e.g. "cnn", "mnist", "imagenet") \n
    `attack`: type of attack performed; "correct" means no attack \n
    
    :return: dict
        Dictionary with keys being the name of the attack variants and values being dictionaries with keys being the name of the techniques and values being the list of grids in that directory.
    """

    # Define load directory based on the dataset, model type and attack
    load_dir = os.path.join(base_dir, dataset, model_type, attack)
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise Exception(f"The directory {load_dir} does not exist.")
    
    # Load grids from all attack variants and techniques
    all_grids = {}
    # Sort the layers, start with the highest (top-level) layers
    # "variants" correspond to the different layers of the model which were randomized
    # "techniques" correspond to the different introspection techniques
    # variants = sorted(os.listdir(kind_path), key=lambda x: int(x.split('_')[0]), reverse=False)
    # for variant in variants:
    #     variant_path = os.path.join(load_dir, variant)
    #     if os.path.isdir(variant_path):
    #         variant_grids = {}
    #         # Note: This part of the code is very similar to the one in load_all_grids, could be refactored if time permits
    #         for technique in os.listdir(variant_path):
    #             technique_path = os.path.join(variant_path, technique)
    #             if os.path.isdir(technique_path):
    #                 grids = load_grids(dataset, model_type, attack, os.path.join(variant, technique), max_images=max_images)
    #                 variant_grids[technique] = grids
    #         all_grids[variant] = variant_grids
            
    # return all_grids

    # kinds = Sequential/Cascading RandomParams
    kinds = sorted(os.listdir(load_dir))

    for kind in kinds:
        kind_path = os.path.join(load_dir, kind)
        if os.path.isdir(kind_path):
            kind_grids = {}

            # Get all variants for this kind
            variants = sorted(os.listdir(kind_path), key=lambda x: int(x.split('_')[0]), reverse=False)

            for variant in variants:
                variant_path = os.path.join(kind_path, variant)
                if os.path.isdir(variant_path):
                    variant_grids = {}

                    # Get all techniques for this variant
                    techniques = os.listdir(variant_path)

                    for technique in techniques:
                        technique_path = os.path.join(variant_path, technique)
                        if os.path.isdir(technique_path):
                            grids = load_grids(dataset, model_type, attack, os.path.join(kind, variant, technique), max_images=max_images)
                            variant_grids[technique] = grids

                    kind_grids[variant] = variant_grids

            all_grids[kind] = kind_grids

    return all_grids

def reduce_dimensionality_umap(all_grids, reducer):
    """
    Reduce the dimensionality of grids using UMAP.
    
    `all_grids`: dict
        Dictionary with keys being the name of the techniques and values being the list of grids.
    `n_components`: int, optional (default=2)
        The number of dimensions to reduce the data to.
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the list of reduced-dimension grids.
    """

    # Initialize a dictionary to store the reduced-dimension grids
    reduced_grids = {}

    # Reduce the dimensionality of each grid
    for technique, grids in all_grids.items():
        print(f"Reducing dimensionality for technique {technique}...")
        # Flatten each grid and stack them into a 2D array
        flat_grids = np.array([grid.flatten() for grid in grids])
        
        # Reduce the dimensionality of the flattened grids
        reduced_grids[technique] = reducer.fit_transform(flat_grids)

    return reduced_grids


def reduce_dimensionality_tsne(all_grids, reducer):
    """
    Reduce the dimensionality of grids using t-SNE.
    
    `all_grids`: dict
        Dictionary with keys being the name of the techniques and values being the list of grids.
    `n_components`: int, optional (default=2)
        The number of dimensions to reduce the data to.
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the list of reduced-dimension grids.
    """

    # Initialize a dictionary to store the reduced-dimension grids
    reduced_grids = {}

    # Reduce the dimensionality of each grid
    for technique, grids in all_grids.items():
        print(f"Reducing dimensionality for technique {technique}...")
        # Flatten each grid and stack them into a 2D array
        flat_grids = np.array([grid.flatten() for grid in grids])
        
        # Reduce the dimensionality of the flattened grids
        reduced_grids[technique] = reducer.fit_transform(flat_grids)

    return reduced_grids

def reduce_dimensionality_kpca(all_grids, reducer):
    """
    Reduce the dimensionality of grids using Kernel PCA.
    
    `all_grids`: dict
        Dictionary with keys being the name of the techniques and values being the list of grids.
    `n_components`: int, optional (default=2)
        The number of dimensions to reduce the data to.
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the list of reduced-dimension grids.
    """

    # Initialize a dictionary to store the reduced-dimension grids
    reduced_grids = {}

    # Reduce the dimensionality of each grid
    for technique, grids in all_grids.items():
        print(f"Reducing dimensionality for technique {technique}...")
        
        # Check if the grids list is empty
        if not grids:
            continue

        # Flatten each grid and stack them into a 2D array
        flat_grids = np.array([grid.flatten() for grid in grids])

        # print(flat_grids)

        try:
            # Reduce the dimensionality of the flattened grids
            reduced_grids[technique] = reducer.fit_transform(flat_grids)
        except ValueError as e:
            # Handle the error gracefully
            print(f'Failed to reduce dimensionality for technique {technique}: {e}')

    return reduced_grids

