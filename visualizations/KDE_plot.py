"""
Plot the accuracy values according to the combination of two hyperparameters
using Kernel Density Estimation (KDE).

The probability distribution of one with respect to the other values is represented as a contour plot
depicting the relationship of the distribution between the two data variables.

- x, y-axis: a pair of hyperparameters (lr/dropout/hidden_unit)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# ------------------------ Load data ----------------------- #
data = pd.read_csv('../results/random_results_80.csv')

hyperparams = ['lr', 'dropout', 'hidden_unit']
accuracy = ['tr_acc', 'val_acc']
title = ['Training Accuracy', 'Validation Accuracy']
hp_pairs = list(itertools.combinations(hyperparams, 2))

# ------------------------ Plot results ----------------------- #
fig, axs = plt.subplots(len(accuracy), len(hyperparams), figsize=(13, 8))

for i, acc in enumerate(accuracy):
    for j, hp in enumerate(hp_pairs):

        '''
        hue: variable that is mapped to determine the color of plot elements.
        fill=True: fill in the area between bivariate contours.
        cbar: add a colorbar to annotate the color mapping in a bivariate plot.
        '''
        sns.kdeplot(data=data, x=hp[0], y=hp[1], hue=acc, ax=axs[i, j],
                    legend=False, fill=True, warn_singular=False, cbar=True)

        options = {
            'title': title[i],
            'xlabel': hp[0],
            'ylabel': hp[1],
        }
        axs[i, j].tick_params(axis='x', rotation=45)
        axs[i, j].set(**options)

plt.tight_layout()
fig.savefig(f"../results/random_KDE.png", dpi=300)  # Save the figure as a PNG file
plt.show()

print('Finished')
