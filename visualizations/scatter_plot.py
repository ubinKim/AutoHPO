import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mtick
from matplotlib.patches import Patch


def scatter_singleHP(data, hyperparams, accuracy):
    """
    Creates a scatter plot for each combination of hyperparameters and accuracy types.

    - x-axis: hyperparameter (lr/dropout/hidden_unit)
    - y-axis: accuracy (training/validation)

    Args:
        data: Pandas DataFrame containing the data.
        hyperparams: List of hyperparameter names (lr/dropout/hidden_unit)
        accuracy: List of accuracy types (training/validation)
    """

    fig, axs = plt.subplots(len(accuracy), len(hyperparams), figsize=(12, 8))

    # Calculate the maximum and minimum accuracy values across all subplots
    y_min = min(min(data[acc]) for acc in accuracy)
    y_max = max(max(data[acc]) for acc in accuracy)

    # Determine the ytick locations and labels for all subplots
    yticks = range(int(y_min * 100), int(y_max * 100) + 1, int((y_max - y_min) * 100 / 6))
    ytick_labels = [int(tick) for tick in yticks]

    for i, acc in enumerate(accuracy):
        acc_percentages = [acc_value * 100 for acc_value in data[acc]]

        for j, hp in enumerate(hyperparams):
            axs[i, j].scatter(data[hp], acc_percentages, alpha=0.5)
            xlabel = axs[i, j].set_xlabel(hp)
            xlabel.set_size(12)
            ylabel = axs[i, j].set_ylabel(acc + ' (%)')
            ylabel.set_size(12)
            axs[i, j].set_yticks(yticks)
            axs[i, j].set_yticklabels(ytick_labels)

    plt.tight_layout()
    fig.savefig(f"../results/pso_scatter_singleHP.png", dpi=300)  # Save the figure as a PNG file


def scatter_twoHP(data, hyperparams, accuracy, title, legend_colors):
    """
    Create a scatter plot for each hyperparameter pairs and color the points based on the accuracy value.
    Divides the range of accuracy into three parts and assigns a different color to each part.

    - x, y-axis: hyperparameter (lr/dropout/hidden_unit)
    - legend (color): accuracy (training/validation)

    Args:
        data: Pandas DataFrame containing the data.
        hyperparams: List of hyperparameter names (lr/dropout/hidden_unit)
        accuracy: List of accuracy types (training/validation)
        title: List of accuracy types (in full name)
        legend_colors: List of colors for the legend
    """

    hp_pairs = list(itertools.combinations(hyperparams, 2))
    fig, axs = plt.subplots(len(accuracy), len(hp_pairs), figsize=(13, 8))

    # Divide the range of accuracy into three parts
    legend_range = np.linspace(np.min(data[accuracy]), np.max(data[accuracy]), num=4)

    for i, acc in enumerate(accuracy):
        # Assign colors based on accuracy values
        acc_colors = []
        for acc_value in data[acc]:
            if acc_value <= legend_range[1]:
                acc_colors.append(legend_colors[0])
            elif acc_value <= legend_range[2]:
                acc_colors.append(legend_colors[1])
            else:
                acc_colors.append(legend_colors[2])

        for j, hp in enumerate(hp_pairs):

            options = {
                'title': title[i],
                'xlabel': hp[0],
                'ylabel': hp[1],
            }
            axs[i, j].scatter(data[hp[0]], data[hp[1]], c=acc_colors, alpha=0.6)
            axs[i, j].set(**options)

            # Add a legend only once, next to the last subplot in the first row
            if j == len(hp_pairs) - 1 and i == 0:
                legend_handles = [Patch(color=color, alpha=0.6) for color in legend_colors]
                legend_labels = ['{:.0f}% - {:.0f}%'.format(legend_range[i] * 100, legend_range[i + 1] * 100)
                                 for i in range(3)]
                axs[i, j].legend(handles=legend_handles, labels=legend_labels, title=title[i],
                                 bbox_to_anchor=(1, 1), loc='upper left')

    plt.tight_layout()
    fig.savefig(f"../results/pso_scatter_twoHP.png", dpi=300)


def scatter_threeHP(data, hyperparams, accuracy, title):
    """
    Create a 3D scatter plot with axes for each hyperparameter and colored based on accuracy values.

    - x, y, z-axis: hyperparameter (lr/dropout/hidden_unit)
    - legend (color): accuracy (training/validation)

    Args:
        data: Pandas DataFrame containing the data.
        hyperparams: List of hyperparameter names (lr/dropout/hidden_unit)
        accuracy: List of accuracy types (training/validation)
        title: List of accuracy types (in full name)
    """

    fig, axs = plt.subplots(1, len(accuracy), figsize=(15, 7), subplot_kw={'projection': '3d'})

    for i, acc in enumerate(accuracy):
        options = {
            'title': title[i],
            'xlabel': hyperparams[0],
            'ylabel': hyperparams[1],
            'zlabel': hyperparams[2]
        }

        # Plot the data points with assigned colors based on accuracy values
        axs[i].scatter(data[hyperparams[0]], data[hyperparams[1]], data[hyperparams[2]], c=data[acc], cmap='viridis')
        axs[i].set(**options)

        # [Optional] Set the initial viewing angles
        # elev: the elevation angle (vertical rotation) | azim: the azimuth angle (horizontal rotation)
        # axs[i].view_init(elev=25, azim=40)

        # Create a colorbar
        norm = plt.Normalize(data[acc].min(), data[acc].max())
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # Set an empty array to be mapped by the colorbar
        formatter = mtick.FuncFormatter(lambda x, _: '{:.0%}'.format(x))
        fig.colorbar(mappable=sm, format=formatter, ax=axs[i], shrink=0.5, pad=0.1, label='accuracy')  # Add a colorbar

    plt.tight_layout()
    fig.savefig(f"../results/pso_scatter_threeHP.png", dpi=300)  # Save the figure as a PNG file


# --------------------------- Load data --------------------------- #
# Load the data
data = pd.read_csv('../results/pso_all_particles_epoch80.csv')

# Define the variables
hyperparams = ['lr', 'dropout', 'hidden_unit']
accuracy = ['tr_acc', 'val_acc']
title = ['Training Accuracy', 'Validation Accuracy']
legend_colors = ['C0', 'C1', 'C2']

# Create the scatter plot
scatter_singleHP(data, hyperparams, accuracy)
scatter_twoHP(data, hyperparams, accuracy, title, legend_colors)
scatter_threeHP(data, hyperparams, accuracy, title)

print("Figures saved successfully!")
