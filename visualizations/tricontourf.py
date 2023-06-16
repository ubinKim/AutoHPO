"""
Directly supply the unordered, irregularly spaced coordinates to Tricontourf (filled contours).

- x, y-axis: hyperparameter (lr/dropout/hidden_unit)
- legend (color): accuracy (training/validation)
"""

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

data = pd.read_csv('../results/random_results.csv')

hyperparams = ['lr', 'dropout', 'hidden_unit']
accuracy = ['tr_acc', 'val_acc']

fig, axs = plt.subplots(len(accuracy), len(hyperparams), figsize=(13, 8))
hp_pairs = list(itertools.combinations(hyperparams, 2))  # all possible pair of hyperparameters

for i, acc in enumerate(accuracy):
    for j, hp in enumerate(hp_pairs):

        # Flatten the values arrays to 1D
        x_values = data[hp[0]].values.flatten()
        y_values = data[hp[1]].values.flatten()
        z_values = data[acc].values.flatten()
        levels = np.linspace(z_values.min(), z_values.max(), 8)

        plt.style.use('_mpl-gallery-nogrid')
        axs[i, j].plot(x_values, y_values, 'o', markersize=2, color='grey')
        cntr = axs[i, j].tricontourf(x_values, y_values, z_values, levels=levels)
        fig.colorbar(cntr, ax=axs[i, j])

        options = {
            'title': f"{acc}",
            'xlabel': f"{hp[0]}",
            'ylabel': f"{hp[1]}",
        }
        axs[i, j].set(**options)

plt.tight_layout()
fig.savefig(f"../results/random_tricontourf.png", dpi=300)  # Save the figure as a PNG file
plt.show()

print('Finished')
