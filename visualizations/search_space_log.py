"""
Creates a search space, one row for each AutoHPO algorithm.

- x, y-axis: hyperparameter (lr/dropout/hidden_unit)
"""

import matplotlib.pyplot as plt
import itertools
import pandas as pd
import math

# --------------------------- Load data --------------------------- #
param_grid = {
    'lr': [0.0001, 0.001, 0.01, 0.1],
    'dropout': [0.1, 0.17, 0.29, 0.5],
    'hidden_unit': [32, 64, 128, 256]
}

random_result_file = pd.read_csv('../results/random_results.csv')
pso_result_file = pd.read_csv('../results/pso_all_particles.csv')

# ----------------- Create a 3x3 grid of subplots ----------------- #
fig, axs = plt.subplots(3, 3, figsize=(12, 10))

plot_colors = ['ro', 'go', 'bo']
algorithms = ['Grid Search', 'Random Search', 'PSO']
hyperparams = ['lr', 'dropout', 'hidden_unit']
hp_pairs_names = list(itertools.combinations(hyperparams, 2))    # all possible pair of hyperparameters

# Set margins as 0.2 times the interval between each grid point ensuring that points are not cut off.
margin = 0.2

for i, hp in enumerate(hp_pairs_names):

    # ----------------------- Plot points ------------------------ #
    # GS
    grid_xlist = param_grid[hp[0]]
    grid_ylist = param_grid[hp[1]]
    grid_points = list(itertools.product(grid_xlist, grid_ylist))
    for gp in grid_points:
        axs[0, i].plot(*gp, plot_colors[i], alpha=0.5)    # 0 (fully transparent) < alpha < 1 (fully opaque)

    # RS
    random_xlist = random_result_file[hp[0]]
    random_ylist = random_result_file[hp[1]]
    random_points = list(zip(random_xlist, random_ylist))
    for rp in random_points:
        axs[1, i].plot(*rp, plot_colors[i], alpha=0.5)

    # PSO
    pso_xlist = pso_result_file[hp[0]]
    pso_ylist = pso_result_file[hp[1]]
    pso_points = list(zip(pso_xlist, pso_ylist))
    for pp in pso_points:
        axs[2, i].plot(*pp, plot_colors[i], alpha=0.5)

    # ------------------------ Plot options ----------------------- #
    x_interval = grid_xlist[1] / grid_xlist[0]
    y_interval = grid_ylist[1] / grid_ylist[0]

    options = {
        'aspect': 'equal',  # make the spacing between points remains consistent along both axes.
        'xlabel': hp[0],
        'xticks': grid_xlist,
        'xticklabels': grid_xlist,
        'xlim': (x_interval ** (math.log(grid_xlist[0], x_interval) - margin),
                 x_interval ** (math.log(grid_xlist[-1], x_interval) + margin)),
        'ylabel': hp[1],
        'yticks': grid_ylist,
        'yticklabels': grid_ylist,
        'ylim': (y_interval ** (math.log(grid_ylist[0], y_interval) - margin),
                 y_interval ** (math.log(grid_ylist[-1], y_interval) + margin))
    }

    for k in range(3):
        axs[k, i].set_xscale('log', base=x_interval)    # log transformations must be done before set(**options)
        axs[k, i].set_yscale('log', base=y_interval)
        axs[k, i].set_title(algorithms[k])
        axs[k, i].set(**options)
        axs[k, i].grid(True)
        axs[k, i].minorticks_off()  # Turn off minor ticks

plt.tight_layout()
plt.subplots_adjust(left=0.25, right=0.75, hspace=0.4)
plt.gcf().set_size_inches(15, 7.5)  # Adjust the figure size
fig.savefig(f"../results/search_space_all.png", dpi=300)  # Save the figure as a PNG file
plt.show()

print('Finished')
