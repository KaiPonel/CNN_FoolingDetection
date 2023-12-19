def bar_graph_distance_differences_singular(diff_distances, techniques):
    metrics = list(diff_distances[0][0].keys())

    for metric in metrics:
        metric_diffs = [[cell[metric] for cell in row] for row in diff_distances]
        
        avg_diffs = [np.mean(diff_row) for diff_row in metric_diffs] # Why? Why average? 

        plt.figure(figsize=(8, 6))
        plt.bar(techniques, avg_diffs, align='center', alpha=0.7)
        plt.xlabel('Technique')
        plt.ylabel('Average Difference')
        plt.title(f'Average {metric} difference between Normal and RandLabel')
        plt.show()


def bar_graph_distance_differences_for_single_metric(diff_distances, techniques, metric):
    # metrics = list(diff_distances[0][0].keys())
    metric_diffs = [[cell[metric] for cell in row] for row in diff_distances]
    for technique, metric_diff in zip(techniques, metric_diffs):
        plt.figure(figsize=(8, 6))
        plt.bar(techniques, metric_diff, align='center', alpha=0.7)
        plt.xlabel('Technique (Versus: {})'.format(technique))
        plt.ylabel('Difference: {}'.format(metric))
        plt.title(f'Difference in {metric} between Normal and RandLabel for {technique}')
        plt.show()

    


def bar_graph_distance_differences_grouped(diff_distances, techniques):
    metrics = list(diff_distances[0][0].keys())

    bar_width = 0.15  
    index = np.arange(len(techniques))  
    opacity = 0.7

    for metric in metrics:
        metric_diffs = [[cell[metric] for cell in row] for row in diff_distances]
        
        avg_diffs = [np.mean(diff_row) for diff_row in metric_diffs]
        
        plt.figure(figsize=(25, 10))
        for i, technique in enumerate(techniques):
            plt.bar(index + i * bar_width, [diff_row[i] for diff_row in metric_diffs], bar_width,
                    alpha=opacity, label=f'{technique} ({avg_diffs[i]:.2f})')  # Display average difference on the legend

        plt.xlabel('Techniques')
        plt.ylabel('Difference')
        plt.title(f'Differences in {metric} between Normal and RandLabel')
        plt.xticks(index + bar_width * (len(techniques) - 1) / 2, techniques)  # Position x-ticks at the center of grouped bars
        plt.legend()
        plt.tight_layout()

        plt.show()


def plot_comparison(data1, data2, introspection_techniques, distance_functions, plot_type='heatmap', use_diverging=False):
    """
    Plots a comparison of distances between introspection techniques using specified data.
        
    Parameters:
        'data1': Distances between introspection techniques for normal inputs.
        'data2': Distances between introspection techniques for UMAP reduced inputs.
        `introspection_techniques` (list): List of introspection techniques.
        `plot_type` (str): Type of plot ('heatmap', 'boxplot', or 'scatter') (default='heatmap').
        `use_diverging` (bool): Indicates whether to use a diverging color map (default=False).
        
    :return: None
    """

    if plot_type == 'heatmap':
        plot_comparison_heatmaps(data1, data2, introspection_techniques, distance_functions, use_diverging=use_diverging)
    
    elif plot_type == 'scatter':
        plot_comparison_scatterplot(data1, data2, introspection_techniques, distance_functions, use_diverging=use_diverging)