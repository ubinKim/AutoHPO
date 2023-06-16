"""
Create a set of annotated heatmaps with a 2 x 3 grid layout.
Each heatmap displays the accuracy (training/validation) which depends on a pair of hyperparameters.

- x, y-axis: a pair of hyperparameters (lr/dropout/hidden_unit)
- cell value: corresponding accuracy (%) (1st row: training / 2nd row: validation)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

data = pd.read_csv('../results/grid_results.csv')
hyperparams = ['lr', 'dropout', 'hidden_unit']
accuracy = ['tr_acc', 'val_acc']

fig, axs = plt.subplots(len(accuracy), len(hyperparams), figsize=(12, 7))
hp_pairs = list(itertools.combinations(hyperparams, 2))     # all possible pair of hyperparameters

for i, acc in enumerate(accuracy):
    for j, hp in enumerate(hp_pairs):
        # Preprocessing: create pivot tables without duplicates
        highest_val_acc = data.groupby(list(hp))[acc].max().reset_index()
        pivot_table = highest_val_acc.pivot(index=hp[1], columns=hp[0], values=acc)
        pivot_table = pivot_table.iloc[::-1]  # Reverse the order of rows

        # Plot the heatmap: display the accuracy as percentages
        hm = sns.heatmap(pivot_table * 100, linewidths=.2, annot=True, fmt='.2f', ax=axs[i, j])
        axs[i, j].set_xlabel(hp[0], size=12)
        axs[i, j].set_ylabel(hp[1], size=12)

        # Add labels to the colorbar
        cbar = hm.collections[0].colorbar
        cbar.set_label(f'{acc} (%)', size=12)

plt.tight_layout()
plt.subplots_adjust(hspace=0.25, wspace=0.35)
fig.savefig(f"../results/grid_heatmap.png", dpi=300)  # Save the figure as a PNG file
plt.show()
