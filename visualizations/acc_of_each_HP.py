"""
Plot changes in validation accuracy according to each hyperparameter using a line plot
to examine the effect of each hyperparameter on the model performance.

- utilizing the results obtained from "grid_search" function
- x-axis: hyperparameter (lr/dropout/hidden_unit)
- y-axis: validation accuracy of each hyperparameter set
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import itertools

# ------------------------ Load data ----------------------- #
data = pd.read_csv('../results/grid_results.csv')

hyperparams = ['lr', 'dropout', 'hidden_unit']
accuracy = ['tr_acc', 'val_acc']
param_grid = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'dropout': [0.1, 0.17, 0.29, 0.5],
    'hidden_unit': [32, 64, 128, 256]
}
hp_pairs_names = list(itertools.combinations(hyperparams, 2))    # all possible pair of hyperparameter names

# ------------------------ Plot results ----------------------- #
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

for i, hp in enumerate(hp_pairs_names):
    grid_fhp1 = param_grid[hp[0]]
    grid_fhp2 = param_grid[hp[1]]
    grid_fixed = list(itertools.product(grid_fhp1, grid_fhp2)) # all possible pair of hyperparameter values (for two HP)
    varying_hp = list(set(hyperparams) - set(hp))[0]
    xticks = param_grid[varying_hp]

    for (fhp1, fhp2) in grid_fixed:
        # Filter the data for specific combinations of two hyperparameters
        filtered_data = data[(data[hp[0]] == fhp1) & (data[hp[1]] == fhp2)]
        x = filtered_data[varying_hp]
        y = filtered_data['val_acc']
        axs[i].plot(x, y*100, label=f'{hp[0]}={fhp1}, {hp[1]}={fhp2}')

    options = {
        'title': f'Accuracy vs. {varying_hp} with Fixed {hp[0]} and {hp[1]}',
        'xlabel': varying_hp,
        'xticks': xticks,
        'xticklabels': xticks,
        'ylabel': 'Accuracy (%)',
    }

    x_interval = xticks[1] / xticks[0]
    axs[i].set_xscale('log', base=x_interval)  # log transformations must be done before set(**options)
    axs[i].set(**options)
    axs[i].grid(True)
    axs[i].minorticks_off()  # Turn off minor ticks
    mplcursors.cursor(axs[i], hover=True)  # Add a cursor to display line information on hover

plt.tight_layout()
plt.gcf().set_size_inches(15, 7.5)  # Adjust the figure size
fig.savefig(f"../results/acc_of_each_HP.png", dpi=300)  # Save the figure as a PNG file
plt.show()

print('Finished')
