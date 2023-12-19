import numpy as np
import os
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from scipy.stats import zscore, norm, ks_2samp
import pandas as pd
# import matplotlib.pyplot as plt


"""
    This part contains the functions to calculate the (static) distance between two images.
    Every function takes two images as input and returns a float value.
"""

def euclidean_distance(im1, im2):
    # flatten both images
    return distance.euclidean(im1, im2)

def manhattan_distance(im1, im2):
    return distance.cityblock(im1, im2)
  
def minkowski_distance(im1, im2, p=3): # p=1 <-> Manhatten, p=2 Euclidian
    return distance.minkowski(im1, im2, p)

def chebyshev_distance(im1, im2):
    return distance.chebyshev(im1, im2)

def cosine_sim(im1, im2):
    return cosine_similarity(im1.reshape(1, -1), im2.reshape(1, -1))

# Dict of all distance functions with their keys being the name of the distance function
# These functions work for both grayscale and RGB images wthout the need of additional parameters
simple_distance_functions = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance,
    "minkowski": minkowski_distance,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_sim,
    #"ssim": ssim
}
# Simple distance function keys + ssim
all_distance_functions_keys = list(simple_distance_functions.keys()) + ["ssim"]



def get_all_simple_distances(im1, im2, use_ssim=False):
    """
        Runs all distance functions on two (flattened) images and returns a dict with the distance functions as keys and the distances as values
        `im1`: first image \n
        `im2`: second image \n

        :return: dict containing the distance functions as keys and the distances as values
    """
    simple = {key: func(im1.flatten(), im2.flatten()) for key, func in simple_distance_functions.items()}
    if use_ssim:
        #create copies of im1 and im2
        im1 = im1.copy()
        im2 = im2.copy()
        try:
            ssim_calc = {"ssim": ssim(im1, im2, channel_axis=2, data_range=max(im1.max(), im2.max()) - min(im1.min(), im2.min()))}
        except:
            ssim_calc = {"ssim": ssim(im1, im2, channel_axis=None, data_range=max(im1.max(), im2.max()) - min(im1.min(), im2.min()))}
        
        return {**simple, **ssim_calc}
    else:
        return simple

def get_all_simple_distances_for_list_of_images(IM1, IM2, average_results=True, use_ssim=False):
    """
        Runs all distance functions on two lists of images and returns a dict with the distance functions as keys and the averaged distances as values
        `IM1`: list of first images \n
        `IM2`: list of second images \n

        :return: dict containing the distance functions as keys and the averaged distances as values
    """
    distances = []
    for im1, im2 in zip(IM1, IM2):
        distances.append(get_all_simple_distances(im1, im2, use_ssim=use_ssim))
    if not average_results:
        return {key: [d[key] for d in distances] for key in distances[0].keys()}
    return {key: np.mean([d[key] for d in distances]) for key in distances[0].keys()}


def test_all_introspection_technique_combinations(grids, print_results=True, average_results=True, use_ssim=False):
    """
        Runs all introspection techniques on a certain model and compares the results using the distance functions defined in simple_distance_functions. \n
        `grids` dict containing the introspection techniques as keys and the grids as values \n
        `average_results` if True, the results of the distance functions are averaged over all images. If False all distance values are being returned for a technique \n
        `print_results`: if True, print the results as a table \n
        `use_ssim`: If True, use ssim as a distance function in addition to the other distance functions

        :return: 2d array containing the distances between the introspection techniques in the following form:
        [
            [distance between technique 1 and technique 1, distance between technique 1 and technique 2, ...],
            [distance between technique 2 and technique 1, distance between technique 2 and technique 2, ...],
            ...
        ]

        Where the distances are dicts containing the distance functions as keys and the distances as values
    """

    if average_results and print_results:
        raise Exception("Cannot print results if `average_results` is True")

    distances = []
    for technique1 in grids.keys():
        for technique2 in grids.keys():
            distances.append(get_all_simple_distances_for_list_of_images(grids[technique1], grids[technique2], average_results=average_results, use_ssim=use_ssim))

    # convert distances into a square (and symmetic) matrix
    distances = np.array(distances).reshape(len(grids.keys()), len(grids.keys()))

    if use_ssim:
        distance_functions = all_distance_functions_keys
    else:
        distance_functions = list(simple_distance_functions.keys())

    # print the results as a table if print_results is True using a library
    if print_results:
        print_all_introspection_technique_combinations(distances=distances, introspection_techniques=grids.keys(), distance_functions=distance_functions)

    return distances, grids.keys(), distance_functions


def get_values_from_distance_matrix(distances, technique1, technique2, distance_functions):
    """
        Extracts the values from the distance matrix for two techniques and a list of distance functions. \n
        `distances`: 2d array containing the distances between the introspection techniques \n
        `technique1`: first technique \n
        `technique2`: second technique \n
        `distance_functions`: list of distance functions \n

        :return: list of distances for the two techniques and the distance functions
    """
    return [distances[technique1][technique2][distance_function] for distance_function in distance_functions]


def extract_single_distance_function_from_distances(distances, distance_function):
    """
        Extracts a single distance function from the distances array. \n
        `distances`: 2d array containing the distances between the introspection techniques \n
        `distance_function`: distance function to extract from the distances array \n

        :return: 2d array containing the distances between the introspection techniques in the following form:
        [
            [distance between technique 1 and technique 1, distance between technique 1 and technique 2, ...],
            [distance between technique 2 and technique 1, distance between technique 2 and technique 2, ...],
            ...
        ]

        Where the distances are dicts containing the distance functions as keys and the distances as values
    """
    return np.array([[cell[distance_function] for cell in row] for row in distances])

def print_all_introspection_technique_combinations(distances, introspection_techniques, distance_functions):
    """
        Prints the results of the test_all_introspection_technique_combinations function as a table. \n
        `distances`: 2d array containing the distances between the introspection techniques \n
        `introspection_techniques`: list of introspection techniques \n
        `distance_functions`: list of distance functions \n

        :return: None
    """

    # print one dataframe for each distance function. The values of the dataframe only contain the respective distance function
    for distance_function in distance_functions:
        print("-" * 100)
        print(f"Distance function: {distance_function}")
        # create a dataframe from the distances array, both the columns and the index are the introspection techniques, the values are the distances between the two techniques
        df = pd.DataFrame(distances, columns=introspection_techniques, index=introspection_techniques)
        df = df.applymap(lambda x: x[distance_function])
        print(df)
        print("-" * 100)

def calc_distributions(grids):
    """
    Calculate distributions for each technique.
    
    `grids`: output from `load_all_grids` function
    
    :return: dict
        Dictionary with keys being the name of the techniques and values being the fitted distributions.
    """
    
    # Initialize dictionary to store distributions
    distributions = {}
    
    # Iterate over each technique
    for technique, grid_list in grids.items():
        # Flatten the grid and calculate distribution
        flattened_grids = [grid.reshape(grid.shape[0], -1) for grid in grid_list]
        flattened_grids = np.concatenate(flattened_grids, axis=0)
        
        # Fit a Gaussian distribution to the data
        mu = np.mean(flattened_grids, axis=0)
        std = np.std(flattened_grids, axis=0)
        
        # Store the distribution parameters
        distributions[technique] = {'mu': mu, 'std': std}
    
    return distributions

def difference_distances(reference, delta, print_results=False, fold_change_and_log_scale=False):
    diff_distances = []
    for i in range(len(reference)):
        diff_row = []
        for j in range(len(reference[i])):
            diff_metrics = {}
            for metric in reference[i][j].keys():
                if fold_change_and_log_scale:
                    # Convert lists to numpy arrays
                    ref_array = np.array(reference[i][j][metric])
                    delta_array = np.array(delta[i][j][metric])

                    # Avoid division by zero
                    mask = ref_array != 0
                    ratio = np.zeros_like(ref_array, dtype=float)
                    ratio[mask] = delta_array[mask] / ref_array[mask]

                    # Apply logarithmic scale with base 2
                    diff_metrics[metric] = np.log2(ratio, out=np.zeros_like(ratio), where=ratio>0)
                else:
                    diff_metrics[metric] = delta_array - ref_array
            diff_row.append(diff_metrics)
        diff_distances.append(diff_row)

    if print_results:
        for row in diff_distances:
            print(row)

    return diff_distances

def apply_log_to_distances(distances):
    """
    Applies a logarithmic scale with base 2 to all values in a list of distances.

    :param distances: List of distances.
    :return: List of distances with logarithmic scale applied.
    """
    log_distances = []
    for i in range(len(distances)):
        log_row = []
        for j in range(len(distances[i])):
            log_metrics = {}
            for metric in distances[i][j].keys():
                # Convert lists to numpy arrays
                distance_array = np.array(distances[i][j][metric])

                # Apply logarithmic scale with base 2
                log_metrics[metric] = np.log2(distance_array, out=np.zeros_like(distance_array, dtype=float), where=distance_array>0)
            log_row.append(log_metrics)
        log_distances.append(log_row)

    return log_distances

def ks_test(grids1, grids2):
    """
    Perform the Kolmogorov-Smirnov test to compare two datasets represented as dictionaries of 2D arrays, and calculate the count of likely same and different pairs, and unclear pairs with the given tolerance.

    `grids1`, `grids2`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids (2D arrays).
    
    :return: None
    """

    # Initialize counters for each technique
    count_same_per_technique = {}
    count_different_per_technique = {}
    count_unclear_per_technique = {}

    # Perform KS test for each technique
    for technique, grids_list1 in grids1.items():
        grids_list2 = grids2.get(technique, [])  # Get corresponding grids for the technique

        # Initialize counters for the current technique
        count_same_per_technique[technique] = 0
        count_different_per_technique[technique] = 0
        count_unclear_per_technique[technique] = 0

        # Perform KS test for the current technique
        for grid1, grid2 in zip(grids_list1, grids_list2):
            statistic, p_value = ks_2samp(grid1.flatten(), grid2.flatten())

            # Update counters based on the p-value
            if p_value > 0.052:
                count_same_per_technique[technique] += 1
            elif p_value < 0.048:
                count_different_per_technique[technique] += 1
            else:
                count_unclear_per_technique[technique] += 1

    # Print summary of results
    print("KS Test Summary:")
    for technique, count_same in count_same_per_technique.items():
        print(f"Technique: {technique}")
        print(f"Count likely same: {count_same}")

    print("\n")

    for technique, count_different in count_different_per_technique.items():
        print(f"Technique: {technique}")
        print(f"Count likely different: {count_different}")

    print("\n")

    for technique, count_unclear in count_unclear_per_technique.items():
        print(f"Technique: {technique}")
        print(f"Count unclear: {count_unclear}")
