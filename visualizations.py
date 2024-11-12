import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


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


def plot_all_datasets_results(results):
    datasets = list(results.keys())  # List of dataset names
    sampling_methods = list(next(iter(results.values())).keys())  # List of sampling methods
    num_datasets = len(datasets)
    num_methods = len(sampling_methods)

    # Width of a single bar
    bar_width = 0.075
    # Generate an array of indices for the datasets
    x = np.arange(num_datasets)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(15, 9))

    # Plot each sampling method as a separate group of bars
    for i, method in enumerate(sampling_methods):
        # Get scores for this method across datasets
        scores = [results[dataset][method] for dataset in datasets]
        # Position bars with an offset for each sampling method
        ax.bar(x + i * bar_width, scores, width=bar_width, label=method)

    # Adding labels and title
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=0.5)
    ax.set_title('Sampling Method Results Across Datasets')
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
        plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Average: {avg_value:.2f}')
        y_min = avg_value - 0.05
        y_max = avg_value + 0.05
        plt.ylim(y_min, y_max)

        # Plot the data as a bar chart
        categories = list(dataset.keys())
        values = list(dataset.values())
        plt.bar(categories, values, color='orange')

        # Add titles and labels
        plt.title(f'Bar Chart for {dataset_name} dataset')
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.xticks(rotation=45, ha='right')
        # Display the plot
        plt.tight_layout()
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

