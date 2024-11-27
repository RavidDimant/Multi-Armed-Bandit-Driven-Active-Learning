import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


PLOT_COLORS = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "black"]


def generate_plot(accuracy_scores_dict, title=''):
    """
    Generate a plot with specified figure size
    """
    plt.figure(figsize=(15, 9))  # Adjust width and height as desired
    for criterion, accuracy_scores in accuracy_scores_dict.items():
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.show()


def generate_bar(dict_scores, title=''):
    """
    Generate a bar plot with specified figure size and properly align long labels
    """
    plt.figure(figsize=(15, 6))  # Adjust width and height as desired
    labels, values = list(dict_scores.keys()), list(dict_scores.values())
    bars = plt.bar(labels, values)

    # Add values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate: center of the bar
            height,  # Y-coordinate: top of the bar
            f'{height:.2f}',  # Format the value to 2 decimal points
            ha='center', va='bottom', fontsize=10, color='black'
        )

    # Rotate and align x-axis labels
    plt.xlabel('Sampling Method')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate labels and align to the right
    plt.ylabel('Aggregated Accuracy Improvement (%)')
    plt.title(title)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Optional: Add grid for clarity
    plt.show()




def plot_all_datasets_results(results, title='', xlabel='', ylabel=''):
    datasets = list(results.keys())  # List of dataset names
    sampling_methods = list(next(iter(results.values())).keys())  # List of sampling methods
    num_datasets = len(datasets)
    num_methods = len(sampling_methods)

    # Width of a single bar
    bar_width = 0.075
    # Generate an array of indices for the datasets
    x = np.arange(num_datasets)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot each sampling method as a separate group of bars
    for i, method in enumerate(sampling_methods):
        # Get scores for this method across datasets
        scores = [results[dataset][method] for dataset in datasets]
        # Use colors from PLOT_COLORS, cycling back if there are more methods than colors
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        # Position bars with an offset for each sampling method
        ax.bar(x + i * bar_width, scores, width=bar_width, label=method, color=color)

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (num_methods - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend(title='Sampling Method')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout for readability
    plt.show()


def plot_bars_for_datasets(data_dict):
    # Loop through each dataset in the dictionary
    for dataset_name, dataset in data_dict.items():
        # Create a figure and axis for each dataset
        plt.figure(figsize=(10, 6))
        # Define the y-axis range (average ± 1)
        avg_value = np.mean(list(dataset.values()))
        # plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Average: {avg_value:.4f}')
        plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Average')
        y_min = avg_value - 0.05
        y_max = avg_value + 0.05
        # plt.ylim(y_min, y_max)

        # Plot the data as a bar chart
        categories = list(dataset.keys())
        values = list(dataset.values())
        plt.bar(categories, values, color='orange')

        # Add titles and labels
        plt.ylim(bottom=0)
        plt.title(f'Bar Chart for {dataset_name} dataset')
        plt.xlabel('Categories')
        # plt.ylabel('Values')
        plt.ylabel('Improvement Over Worst (%)')
        plt.xticks(rotation=45, ha='right')
        # Display the plot
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()


def plot_bars_for_methods(data_dict):
    # Prepare data for plotting by converting the dictionary to a flat list of records
    data = []
    for dataset_name, methods in data_dict.items():
        for method_name, score in methods.items():
            data.append({'Method': method_name, 'Dataset': dataset_name, 'Score': score})

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Set up the Seaborn plot
    plt.figure(figsize=(10, 6))
    sns.set_palette("Set2")  # You can adjust the color palette

    # Create the horizontal barplot with sub-bars for each method and dataset
    ax = sns.barplot(x='Score', y='Method', hue='Dataset', data=df, dodge=True, errorbar=None)

    # Add labels and title
    plt.title('Scores of Sampling Methods across Datasets')
    plt.xlabel('Score')
    plt.ylabel('Sampling Method')

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_horizontal_bootstrap(dicts_list, data_names):

    dicts_keys = list(dicts_list[0].keys())




    key_colors = {dicts_keys[i]: PLOT_COLORS[i] for i in range(len(dicts_keys))}

    # Spacing parameters
    bar_spacing = 0.2  # Spacing between bars in a group
    dict_spacing = 0.6  # Larger spacing between groups of dictionaries

    # Compact figure dimensions
    fig_height = max(10, len(dicts_list) * dict_spacing * 1.5)  # Dynamically set height
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Adjust y positions
    y_positions = []
    y_labels = []
    current_y = 0
    for idx, d in enumerate(dicts_list):
        y_positions.append(current_y)
        y_labels.append(f'{data_names[idx]}')
        current_y += len(d) * bar_spacing + dict_spacing - bar_spacing  # Account for bar and dict spacing

    # Plot each dictionary
    for idx, d in enumerate(dicts_list):
        keys = list(d.keys())
        means = [np.mean(d[key]) for key in keys]
        q05 = [np.quantile(d[key], 0.05) for key in keys]
        q95 = [np.quantile(d[key], 0.95) for key in keys]

        # Plot thick line segments for quantile ranges
        for i, key in enumerate(keys):
            y = y_positions[idx] + i * bar_spacing
            color = key_colors[key]  # Get the color for this key
            ax.plot([q05[i], q95[i]], [y, y], color=color, linewidth=5, alpha=0.6)

            # Plot large diamond markers for the mean
            ax.scatter(means[i], y, color=color, marker='D', s=100, label=key if idx == 0 else None)

    # Set labels, legend, and formatting
    ax.set_yticks([pos + (len(dicts_list[0]) - 1) * bar_spacing / 2 for pos in y_positions])  # Center y-ticks for each dictionary
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Value")
    ax.invert_yaxis()  # Flip y-axis for a more natural layout
    ax.legend(title="Keys")
    plt.title("Quantiles and Means with Grouped Dictionary Labels")
    plt.tight_layout()
    plt.show()


def plot_boxplots(sim_res):
    # Assign unique colors to each method across all dictionaries
    methods = set(key for d in sim_res.values() for key in d.keys())
    method_colors = {method: color for method, color in zip(methods, plt.cm.tab10.colors)}
    dict_names = list(sim_res.keys())
    num_methods = len(methods)
    num_dicts = len(sim_res.keys())

    # Prepare data for plotting
    all_results = []
    all_labels = []
    all_positions = []
    y_tick_labels = []

    current_position = 1  # Track the position for each box
    # Loop over all dictionaries to prepare the data for plotting
    for dict_name, results in sim_res.items():
        for method, values in results.items():
            all_results.append(values)
            all_labels.append(method)  # Label the method for each result
            all_positions.append(current_position)
            current_position += 1
        current_position += 1  # Add spacing between dictionaries
        y_tick_labels.append(dict_name)  # Add dictionary name to the y-ticks

    # Create the horizontal boxplot
    plt.figure(figsize=(12, 8))
    box = plt.boxplot(
        all_results,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        vert=False,
        medianprops={"linewidth": 0},  # Hide the median line
        positions=all_positions,
    )

    # Customize boxplot appearance
    for i, patch in enumerate(box['boxes']):
        method = all_labels[i]
        patch.set_facecolor(method_colors[method])  # Set the color for each method

    # Add mean points
    means = [np.mean(values) for values in all_results]
    plt.scatter(means, all_positions, color="red", label="Mean", zorder=3)

    # Add legend: show both methods and mean
    handles = [
        plt.Line2D([0], [0], color="red", marker="o", linestyle="None", label="Mean"),
    ] + [
        plt.Line2D([0], [0], color=color, lw=4, linestyle="None", label=method)
        for method, color in method_colors.items()
    ]
    plt.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.15, 0.5))

    # Formatting


    plt.yticks(
        np.arange((num_methods+1)/2, num_dicts*num_methods + num_dicts-1, num_methods + 1), y_tick_labels, fontsize=10
    )  # Label dictionaries



    plt.xlabel("Simulation Results")
    plt.ylabel("Experiments")
    plt.title("Simulation Results Across Experiments")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Adjust layout and make sure labels are positioned properly
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_dict_boxplots(res_dict, name=''):
    methods = list(res_dict.keys())
    data = list(res_dict.values())

    plt.figure(figsize=(8, 6))
    box = plt.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        vert=False,
        medianprops={"linewidth": 0},  # Hide the median line
    )

    # Customize boxplot appearance
    colors = PLOT_COLORS[:len(methods)]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add mean points
    means = [np.mean(results) for results in data]
    plt.scatter(means, range(1, len(means) + 1), color="black", label="Mean", zorder=3)

    # Add legend with method names and their colors
    # handles = [plt.Line2D([0], [0], color=color, lw=4, label=method) for method, color in zip(methods, colors)]
    # handles.append(plt.Line2D([0], [0], color='black', marker='o', linestyle='None', label='Mean'))
    # handles.reverse()

    handles = [plt.Line2D([0], [0], color='black', marker='o', linestyle='None', label='Mean')]

    # Customize the legend
    plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # Formatting
    plt.yticks(range(1, len(methods) + 1), methods, fontsize=10)
    plt.xlabel("Accuracy")
    # plt.ylabel(name)
    plt.title(name)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.show()
