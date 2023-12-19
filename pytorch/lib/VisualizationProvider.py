import matplotlib.pyplot as plt
from scipy.stats import norm 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal, ks_2samp
from matplotlib import cm


def visualize_grids(grids, n=5, ground_truth_images=None, save_name=None):
    """
        Visualize a grid of images. \n
        `grids`: dict containing a string as a key and an array of images stored in numpy arrays \n
        `n`: number of images to be visualized per key \n
        `ground_truth_images`: list of original images to be displayed at the top row
    """
    # Define the size of the plot
    rows = len(grids.keys()) + (1 if ground_truth_images is not None else 0)
    fig, axs = plt.subplots(rows, n, figsize=(20, 20))
    
    # If ground truth images are provided, display them at the top row
    if ground_truth_images is not None:
        for j, image in enumerate(ground_truth_images[:n]):
            axs[0, j].imshow(image, cmap="gray")
            axs[0, j].axis('off')
            axs[0, j].set_title('Original')
    
    # Iterate over all keys sorted alphabetically
    for i, key in enumerate(sorted(grids.keys())):
        # Get the first n images for the current key
        images = grids[key][:n]
        # Iterate over the images and plot them
        for j, image in enumerate(images):
            row = i + (1 if ground_truth_images is not None else 0)
            axs[row, j].imshow(image, cmap="gray")
            axs[row, j].axis('off')
            axs[row, j].set_title(key)

    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    
    plt.show()




def visualize_grids_randparams(grids, max_rows=5, image_size=5):
    """
        Visualize a grid of images for the RandParams attack. \n
        `grids`: dict containing a string as a key and an array of images stored in numpy arrays \n
        `max_rows`: maximum number of rows to be visualized \n

        The function will visualize the first `max_rows` rows of each key in `grids`.
    """
    # Get number of techniques and variants
    n_techniques = len(next(iter(grids.values())))
    n_variants = len(grids)

    # Visualize grids for each technique for the RandParams attack
    technique_names = list(next(iter(grids.values())).keys())
    for technique in range(n_techniques):
        technique_name = technique_names[technique]
        
        n_grids = min(len(variant_grids[technique_name]) for variant_grids in grids.values()) # Number of rows possible for current technique
        n_rows = min(n_grids, max_rows) # Number of rows to render, based on max_rows
        
        fig, axes = plt.subplots(n_rows, n_variants, figsize=(n_variants * image_size, n_rows * image_size))
        fig.suptitle(f'Technique {technique_name}')
        
        # Visualize grids for current introspection technique
        for i in range(n_rows):
            for j, (variant, variant_grids) in enumerate(grids.items()):
                ax = axes[i, j]
                ax.imshow(variant_grids[technique_name][i], cmap='hot', interpolation='nearest')
                ax.axis('off')
                if i == 0:
                    ax.set_title(variant)
    plt.show()

def plot_heatmap(distances, introspection_techniques, distance_functions, use_diverging=False, color_scale=None, save_name=None):
    """
        Plots the results of the difference_distances function as a heatmap. \n
        `distances`: 2d array containing the distances between the introspection techniques \n
        `introspection_techniques`: list of introspection techniques \n
        `distance_functions`: list of distance functions \n
        `use_diverging`: boolean indicating whether to use a diverging color map
        `color_scale`: tuple indicating the min and max value for the color scale

        :return: None
    """

    # Plot one heatmap for each distance function. The values of the heatmap only contain the respective distance function
    for distance_function in distance_functions:
        print("-" * 100)
        print(f"Distance function: {distance_function}")
        
        # Create a DataFrame from the distances array, both the columns and the index are the introspection techniques, 
        # the values are the distances between the two techniques
        df = pd.DataFrame(distances, columns=introspection_techniques, index=introspection_techniques)
        
        # Select only the values for the current distance function
        df = df.applymap(lambda x: x[distance_function])

        # Create a heatmap
        plt.figure(figsize=(10, 8))
        
        if use_diverging:
            cmap = "coolwarm"
        else:
            cmap = "inferno"
        
        if color_scale is not None:
            sns.heatmap(df, annot=True, cmap=cmap, vmin=color_scale[0], vmax=color_scale[1])
        else:
            sns.heatmap(df, annot=True, cmap=cmap)

        if save_name is not None:
            plt.savefig(save_name, dpi=600)    
        # Show the plot
        plt.show()
        
        print("-" * 100)



def plot_comparison_heatmaps(distances1, distances2, introspection_techniques, distance_functions, use_diverging=False, title_distances1="Distances 1", title_distances2="Distances 2", distances3=None, title_distances3=None, save_name=None):
    """
    Plots the results of the difference_distances function as a heatmap. 
    'normal_diff_distances': Distances between introspection techniques for normal inputs.
    'umap_diff_distances': Distances between introspection techniques for UMAP reduced inputs.
    `introspection_techniques`: list of introspection techniques 
    `distance_functions`: list of distance functions 
    `use_diverging`: boolean indicating whether to use a diverging color map

    :return: None
    """
    
    for distance_function in distance_functions:
        print("-" * 100)
        print(f"Distance function: {distance_function}")
        
        # Create DataFrames from the diff_distances arrays for normal and UMAP reduced inputs
        normal_df = pd.DataFrame(distances1, columns=introspection_techniques, index=introspection_techniques)
        umap_df = pd.DataFrame(distances2, columns=introspection_techniques, index=introspection_techniques)
        
        # Select only the values for the current distance function
        normal_df = normal_df.applymap(lambda x: x[distance_function])
        umap_df = umap_df.applymap(lambda x: x[distance_function])

        # Calculate the min and max values across both dataframes
        min_val = min(normal_df.min().min(), umap_df.min().min())
        max_val = max(normal_df.max().max(), umap_df.max().max())

        if distances3 is not None:
            third_df = pd.DataFrame(distances3, columns=introspection_techniques, index=introspection_techniques)
            third_df = third_df.applymap(lambda x: x[distance_function])
            min_val = min(min_val, third_df.min().min())
            max_val = max(max_val, third_df.max().max())

        # Create subplots for side by side comparison
        if distances3 is not None:
            fig, axes = plt.subplots(1, 3, figsize=(22.5, 8))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        if use_diverging:
            cmap = "coolwarm"
        else:
            cmap = "inferno"

        # Plot normal input heatmap
        sns.heatmap(normal_df, annot=True, cmap=cmap, ax=axes[0], vmin=min_val, vmax=max_val)
        axes[0].set_title(title_distances1)

        # Plot UMAP input heatmap
        sns.heatmap(umap_df, annot=True, cmap=cmap, ax=axes[1], vmin=min_val, vmax=max_val)
        
        axes[1].set_title(title_distances2)

        if distances3 is not None:
            sns.heatmap(third_df, annot=True, cmap=cmap, ax=axes[2], vmin=min_val, vmax=max_val)
            axes[2].set_title(title_distances3)
        
        # Show the subplots
        if save_name is not None:
            plt.savefig(save_name, dpi=600)

        plt.show()
        
        print("-" * 100)

  

def plot_comparison_scatterplot(normal_diff_distances, umap_diff_distances, introspection_techniques, distance_functions, use_diverging=False, title_distances1="Distances 1", title_distances2="Distances 2", save_name=None):
    """
    Plots the results of the difference_distances function as a scatter plot. 
    'normal_diff_distances': Distances between introspection techniques for normal inputs.
    'umap_diff_distances': Distances between introspection techniques for UMAP reduced inputs.
    `introspection_techniques`: list of introspection techniques 
    `distance_functions`: list of distance functions 
    `use_diverging`: boolean indicating whether to use a diverging color map

    :return: None
    """
    for distance_function in distance_functions:
        print("-" * 100)
        print(f"Distance function: {distance_function}")

        # Create DataFrames from the diff_distances arrays for normal and UMAP reduced inputs
        normal_df = pd.DataFrame(normal_diff_distances, columns=introspection_techniques, index=introspection_techniques)
        umap_df = pd.DataFrame(umap_diff_distances, columns=introspection_techniques, index=introspection_techniques)

        # Select only the values for the current distance function
        normal_df = normal_df.applymap(lambda x: x[distance_function])
        umap_df = umap_df.applymap(lambda x: x[distance_function])

        # Create subplots for side by side comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # Plot normal input scatter plot
        for i, technique in enumerate(introspection_techniques):
            x_coords = [i] * len(introspection_techniques)
            y_coords = normal_df[technique].values
            axes[0].scatter(x_coords, y_coords, label=technique)

        axes[0].set_xticks(range(len(introspection_techniques)))
        axes[0].set_xticklabels(introspection_techniques, rotation=90)
        axes[0].set_title(title_distances1)
        axes[0].legend()

        # Plot UMAP input scatter plot
        for i, technique in enumerate(introspection_techniques):
            x_coords = [i] * len(introspection_techniques)
            y_coords = umap_df[technique].values
            axes[1].scatter(x_coords, y_coords, label=technique)

        axes[1].set_xticks(range(len(introspection_techniques)))
        axes[1].set_xticklabels(introspection_techniques, rotation=90)
        axes[1].set_title(title_distances2)
        axes[1].legend()

        # Show the subplots
        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name, dpi=600)
        plt.show()

        print("-" * 100)


# Note: This is used to plot Lower Dimensional (2D) Representations of the grids
def plot_2d_distributions(grids1, grids2, save_name=None):
    """
    Calculate 2D distributions and plot them in 3D for each technique.
    
    `grids1`, `grids2`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids.
    
    :return: None
    """
    
    # Get the list of techniques
    techniques = list(grids1.keys())
    
    # Create subplots
    fig = plt.figure(figsize=(10, len(techniques)*5))
    
    # Iterate over each technique
    for i, technique in enumerate(techniques):
        # Get grids
        grid_list1 = grids1[technique]
        grid_list2 = grids2[technique]
        
        # Fit a 2D Gaussian distribution to the data
        mu1 = np.mean(grid_list1, axis=0)
        cov1 = np.cov(grid_list1.T)
        
        mu2 = np.mean(grid_list2, axis=0)
        cov2 = np.cov(grid_list2.T)
        
       # Define maximum grid size (Higher: More precision, more memory usage + time to calculate)
        max_grid_size = 2000  

        # Calculate ranges
        range_x = max(mu1[0], mu2[0]) - min(mu1[0], mu2[0]) + 6*np.sqrt(max(cov1[0,0], cov2[0,0]))
        range_y = max(mu1[1], mu2[1]) - min(mu1[1], mu2[1]) + 6*np.sqrt(max(cov1[1,1], cov2[1,1]))

        # Calculate step sizes
        step_x = range_x / (max_grid_size + 1e-7)
        step_y = range_y / (max_grid_size + 1e-7)

        # Generate x, y values
        x, y = np.mgrid[min(mu1[0], mu2[0]) - 3*np.sqrt(max(cov1[0,0], cov2[0,0])):max(mu1[0], mu2[0]) + 3*np.sqrt(max(cov1[0,0], cov2[0,0])):step_x,
                        min(mu1[1], mu2[1]) - 3*np.sqrt(max(cov1[1,1], cov2[1,1])):max(mu1[1], mu2[1]) + 3*np.sqrt(max(cov1[1,1], cov2[1,1])):step_y]

        
        # Generate z values (PDF)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        
        Z1 = multivariate_normal(mu1, cov1).pdf(pos)
        Z2 = multivariate_normal(mu2, cov2).pdf(pos)
        
        # Plot the distributions
        ax = fig.add_subplot(len(techniques), 2, 2*i+1, projection='3d')
        ax.plot_surface(x, y, Z1, cmap=cm.RdYlGn)
        ax.set_title(f'{technique} - Dist 1')
        
        ax = fig.add_subplot(len(techniques), 2, 2*i+2, projection='3d')
        ax.plot_surface(x, y, Z2, cmap=cm.RdYlGn)
        ax.set_title(f'{technique} - Dist 2')
    
    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    # Show the plot
    plt.show()


def plot_2d_distributions2(grids1, grids2, grids3, title1="Dist 1", title2="Dist 2", title3="Dist 3", save_name=None):
    """
    Calculate 2D distributions and plot them in 3D for each technique.
    
    `grids1`, `grids2`, `grids3`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids.
    
    :return: None
    """
    
    # Get the list of techniques
    techniques = list(grids1.keys())
    
    # Create subplots
    fig = plt.figure(figsize=(15, len(techniques)*5))
    
    # Iterate over each technique
    for i, technique in enumerate(techniques):
        # Get grids
        grid_list1 = grids1[technique]
        grid_list2 = grids2[technique]
        grid_list3 = grids3[technique]
        
        # Fit a 2D Gaussian distribution to the data
        mu1 = np.mean(grid_list1, axis=0)
        cov1 = np.cov(grid_list1.T)
        
        mu2 = np.mean(grid_list2, axis=0)
        cov2 = np.cov(grid_list2.T)

        mu3 = np.mean(grid_list3, axis=0)
        cov3 = np.cov(grid_list3.T)
        
       # Define maximum grid size (Higher: More precision, more memory usage + time to calculate)
        max_grid_size = 2000  

        # Calculate ranges
        range_x = max(mu1[0], mu2[0], mu3[0]) - min(mu1[0], mu2[0], mu3[0]) + 6*np.sqrt(max(cov1[0,0], cov2[0,0], cov3[0,0]))
        range_y = max(mu1[1], mu2[1], mu3[1]) - min(mu1[1], mu2[1], mu3[1]) + 6*np.sqrt(max(cov1[1,1], cov2[1,1], cov3[1,1]))

        # Calculate step sizes
        step_x = range_x / (max_grid_size + 1e-7)
        step_y = range_y / (max_grid_size + 1e-7)

        # Generate x, y values
        x, y = np.mgrid[min(mu1[0], mu2[0], mu3[0]) - 3*np.sqrt(max(cov1[0,0], cov2[0,0], cov3[0,0])):max(mu1[0], mu2[0], mu3[0]) + 3*np.sqrt(max(cov1[0,0], cov2[0,0], cov3[0,0])):step_x,
                        min(mu1[1], mu2[1], mu3[1]) - 3*np.sqrt(max(cov1[1,1], cov2[1,1], cov3[1,1])):max(mu1[1], mu2[1], mu3[1]) + 3*np.sqrt(max(cov1[1,1], cov2[1,1], cov3[1,1])):step_y]

        
        # Generate z values (PDF)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        
        Z1 = multivariate_normal(mu1, cov1).pdf(pos)
        Z2 = multivariate_normal(mu2, cov2).pdf(pos)
        Z3 = multivariate_normal(mu3, cov3).pdf(pos)
        
        # Plot the distributions
        ax = fig.add_subplot(len(techniques), 3, 3*i+1, projection='3d')
        ax.plot_surface(x, y, Z1, cmap=cm.RdYlGn)
        ax.set_title(f'{technique} - {title1}')
        
        ax = fig.add_subplot(len(techniques), 3, 3*i+2, projection='3d')
        ax.plot_surface(x, y, Z2, cmap=cm.RdYlGn)
        ax.set_title(f'{technique} - {title2}')

        ax = fig.add_subplot(len(techniques), 3, 3*i+3, projection='3d')
        ax.plot_surface(x, y, Z3, cmap=cm.RdYlGn)
        ax.set_title(f'{technique} - {title3}')
    
    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    # Show the plot
    plt.show()


def plot_scatter(grids1, grids2, save_name=None):
    """
    Plot scatter plots side by side for each technique.
    
    `grids1`, `grids2`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids.
    
    :return: None
    """
    
    # Get the list of techniques
    techniques = list(grids1.keys())
    
    # Create subplots
    fig, axs = plt.subplots(len(techniques), 2, figsize=(10, len(techniques)*5))
    
    # Iterate over each technique
    for i, technique in enumerate(techniques):
        # Get grids
        grid_list1 = grids1[technique]
        grid_list2 = grids2[technique]
        
        # Concatenate all grids for each technique
        concatenated_grids1 = np.vstack(grid_list1)
        concatenated_grids2 = np.vstack(grid_list2)
        
        # Get the limits for x and y axes
        x_min = min(concatenated_grids1[:, 0].min(), concatenated_grids2[:, 0].min())
        x_max = max(concatenated_grids1[:, 0].max(), concatenated_grids2[:, 0].max())
        
        y_min = min(concatenated_grids1[:, 1].min(), concatenated_grids2[:, 1].min())
        y_max = max(concatenated_grids1[:, 1].max(), concatenated_grids2[:, 1].max())
        
        # Plot the scatter plots with the same axis scaling
        axs[i, 0].scatter(concatenated_grids1[:, 0], concatenated_grids1[:, 1])
        axs[i, 0].set_xlim([x_min, x_max])
        axs[i, 0].set_ylim([y_min, y_max])
        axs[i, 0].set_title(f'{technique} - Grids 1')
        
        axs[i, 1].scatter(concatenated_grids2[:, 0], concatenated_grids2[:, 1])
        axs[i, 1].set_xlim([x_min, x_max])
        axs[i, 1].set_ylim([y_min, y_max])
        axs[i, 1].set_title(f'{technique} - Grids 2')
    
    # Show the plot
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    plt.show()

def plot_scatter2(grids1, grids2, grids3, save_name=None, title1="Grids 1", title2="Grids 2", title3="Grids 3"):
    """
    Plot scatter plots side by side for each technique.
    
    `grids1`, `grids2`, `grids3`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids.
    
    :return: None
    """
    
    # Get the list of techniques
    techniques = list(grids1.keys())
    
    # Create subplots
    fig, axs = plt.subplots(len(techniques), 3, figsize=(15, len(techniques)*5))
    
    # Iterate over each technique
    for i, technique in enumerate(techniques):
        # Get grids
        grid_list1 = grids1[technique]
        grid_list2 = grids2[technique]
        grid_list3 = grids3[technique]
        
        # Concatenate all grids for each technique
        concatenated_grids1 = np.vstack(grid_list1)
        concatenated_grids2 = np.vstack(grid_list2)
        concatenated_grids3 = np.vstack(grid_list3)
        
        # Get the limits for x and y axes
        x_min = min(concatenated_grids1[:, 0].min(), concatenated_grids2[:, 0].min(), concatenated_grids3[:, 0].min())
        x_max = max(concatenated_grids1[:, 0].max(), concatenated_grids2[:, 0].max(), concatenated_grids3[:, 0].max())
        
        y_min = min(concatenated_grids1[:, 1].min(), concatenated_grids2[:, 1].min(), concatenated_grids3[:, 1].min())
        y_max = max(concatenated_grids1[:, 1].max(), concatenated_grids2[:, 1].max(), concatenated_grids3[:, 1].max())
        
        # Plot the scatter plots with the same axis scaling
        axs[i, 0].scatter(concatenated_grids1[:, 0], concatenated_grids1[:, 1])
        axs[i, 0].set_xlim([x_min, x_max])
        axs[i, 0].set_ylim([y_min, y_max])
        axs[i, 0].set_title(f'{technique} - {title1}')
        
        axs[i, 1].scatter(concatenated_grids2[:, 0], concatenated_grids2[:, 1])
        axs[i, 1].set_xlim([x_min, x_max])
        axs[i, 1].set_ylim([y_min, y_max])
        axs[i, 1].set_title(f'{technique} - {title2}')

        axs[i, 2].scatter(concatenated_grids3[:, 0], concatenated_grids3[:, 1])
        axs[i, 2].set_xlim([x_min, x_max])
        axs[i, 2].set_ylim([y_min, y_max])
        axs[i, 2].set_title(f'{technique} - {title3}')
    
    # Show the plot
    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    plt.show()



def plot_gaussian_distributions(distances, techniques, metric, value_range=None, save_name=None):
    """
    Plots Gaussian distributions for each combination of techniques. (NxN plots)

    :param distances: 2D array of distances between techniques. This is the output of the difference_distances / test_all_introspection_technique_combinations function.
    :param techniques: List of technique names (output of test_all_introspection_technique_combinations Function).
    :param metric: The distance metric to use (output of test_all_introspection_technique_combinations Function). 
    :param value_range: Tuple consisting of two values (min, max) describing the min and max x values for the plot. If not passed it should default to None.
    """
    num_techniques = len(techniques)
    fig, axs = plt.subplots(num_techniques, num_techniques, figsize=(15, 15))

    # Initialize min and max values for x-axis
    min_val = float('inf')
    max_val = float('-inf')

    # Create a Gaussian distribution for each pair of techniques and find the global min and max
    gaussian_distributions = []
    for i in range(num_techniques):
        for j in range(num_techniques):
            # Extract the distances for the current pair of techniques
            technique_distances = distances[i][j][metric]

            # Create a Gaussian distribution from the distances
            mu, std = np.mean(technique_distances), np.std(technique_distances)
            gaussian_distribution = np.random.normal(mu, std, 1000)
            gaussian_distributions.append(gaussian_distribution)

            # Update min and max values
            min_val = min(min_val, gaussian_distribution.min())
            max_val = max(max_val, gaussian_distribution.max())

    # If value_range is provided, use it instead of calculated min and max values
    if value_range is not None:
        min_val, max_val = value_range

    # Plot each Gaussian distribution with the same x-axis scale
    for i in range(num_techniques):
        for j in range(num_techniques):
            # Plot the distribution
            sns.histplot(gaussian_distributions[i * num_techniques + j], ax=axs[i, j], kde=True)

            # Set the title of the subplot to the names of the techniques
            axs[i, j].set_title(f"{techniques[i]} vs {techniques[j]}")

            # Set the same x-axis limits for all subplots
            axs[i, j].set_xlim(min_val, max_val)

    # Set the labels for the rows and columns to the technique names
    cols = [ax.set_xlabel(technique) for ax, technique in zip(axs[0], techniques)]
    rows = [ax.set_ylabel(technique) for ax, technique in zip(axs[:,0], techniques)]

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    plt.show()



def qq_plot(grids1, grids2):
    """
    Create Q-Q plots to compare two datasets represented as dictionaries of 2D arrays.
    
    `grids1`, `grids2`: dict
        Dictionaries with keys being the name of the techniques and values being the list of grids (2D arrays).
    
    :return: None
    """
    # Create Q-Q plots for each technique
    for technique, grids_list1 in grids1.items():
        grids_list2 = grids2.get(technique, [])  # Get corresponding grids for the technique
        
        # Calculate quantiles for grids in each technique
        quantiles1 = np.percentile(np.vstack(grids_list1), np.arange(1, 100))
        quantiles2 = np.percentile(np.vstack(grids_list2), np.arange(1, 100))
        
        # Create Q-Q plot for the current technique
        plt.figure(figsize=(8, 6))
        plt.scatter(quantiles1, quantiles2)
        plt.xlabel(f'Quantiles of {technique} - Grids 1')
        plt.ylabel(f'Quantiles of {technique} - Grids 2')
        plt.title(f'Q-Q Plot - {technique}')
        
        # Add a reference line (y=x)
        min_val = min(quantiles1.min(), quantiles2.min())
        max_val = max(quantiles1.max(), quantiles2.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
        
        # Show the plot for the current technique
        plt.show()

def compare_mean_stddev(grids1, grids2):
    techniques = []
    mean1_values = []
    std_dev1_values = []
    mean2_values = []
    std_dev2_values = []
    
    # Calculate mean and std_dev for each technique in grids1
    for technique, grids in grids1.items():
        mean1 = np.mean(grids)
        std_dev1 = np.std(grids)
        techniques.append(technique)
        mean1_values.append(mean1)
        std_dev1_values.append(std_dev1)
    
    # Calculate mean and std_dev for each technique in grids2
    for technique, grids in grids2.items():
        mean2 = np.mean(grids)
        std_dev2 = np.std(grids)
        mean2_values.append(mean2)
        std_dev2_values.append(std_dev2)
    
    # Plotting
    x = np.arange(len(techniques))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, mean1_values, width, label='Mean - Grids1')
    rects2 = ax.bar(x + width/2, mean2_values, width, label='Mean - Grids2')
    
    ax.set_ylabel('Mean Values')
    ax.set_title('Comparison of Mean Values for Different Techniques')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation='vertical')
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    # Plotting standard deviations
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, std_dev1_values, width, label='Std Dev - Grids1')
    rects2 = ax.bar(x + width/2, std_dev2_values, width, label='Std Dev - Grids2')
    
    ax.set_ylabel('Standard Deviation Values')
    ax.set_title('Comparison of Standard Deviation Values for Different Techniques')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation='vertical')
    ax.legend()

    plt.tight_layout()
    plt.show()


