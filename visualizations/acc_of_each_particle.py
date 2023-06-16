"""
Plot the validation accuracy of each hyperparameter set over the epochs using a line plot
to examine the effect of number of epochs on the model performance.

To differentiate which line corresponds to which hyperparameter set,
display the information of a line when hovering the mouse over it: hyperparameter set, epoch (x), accuracy (y).

- utilizing the results obtained from "grid_with_epoch" function
- x-axis: the number of epochs
- y-axis: validation accuracy of each hyperparameter set
- legend: hyperparameter set
"""

import pandas as pd
import matplotlib.pyplot as plt
import mplcursors

# ------------------------ Load data ----------------------- #
data = pd.read_csv('../results/grid_with_epoch_results.csv')

# Extract the values from the first column
hp_names = data.iloc[:, 0].tolist()    # .tolist(): convert the selected column to a list

# Extract the values from the rest of the columns (row ~ hyperparameter set | column ~ #epoch)
acc_per_epoch = data.iloc[:, 1:]*100

# Transpose the DataFrame (NOW: row ~ #epoch | column ~ hyperparameter set)
acc_per_hps = acc_per_epoch.T

# ------------------------ Plot results ----------------------- #
options = {
    'title': 'Impact of Number of Epochs on Different Hyperparameter Sets',
    'xlabel': 'Epoch',
    'xticks': [0, 9, 19, 29, 39, 49, 59, 69, 79],
    'ylabel': 'Accuracy (%)',
}

fig, ax = plt.subplots()

# Plot the accuracy of each hyperparameter set over the epochs
for name, hps in zip(hp_names, acc_per_hps.columns):
    ax.plot(acc_per_hps[hps], label=f'{name}')

ax.set(**options)
plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.1)
plt.gcf().set_size_inches(10, 7)  # Adjust the figure size
mplcursors.cursor(ax, hover=True)   # Add a cursor to display line information on hover
fig.savefig(f"../results/acc_of_each_particle.png", dpi=800)  # Save the figure as a PNG file
plt.show()

print('Finished')
