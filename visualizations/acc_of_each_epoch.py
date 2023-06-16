"""
Plot the trend of the validation accuracy over the epochs using a line plot
to examine the effect of number of epochs on the model performance.

- applied descriptive statistics: mean/median/min/max/IQR25%/IQR75%
- utilizing the results obtained from "grid_with_epoch" function
- x-axis: the number of epochs
- y-axis: validation accuracy of each hyperparameter set
"""

import pandas as pd
import matplotlib.pyplot as plt
import csv

# ------------------------ Load data ----------------------- #
no_of_epoch = 80    # columns

# Read the CSV file without the first column (~ header)
data = pd.read_csv('../results/grid_with_epoch_results.csv', usecols=range(1, no_of_epoch+1))

# Calculate mean/median/min/max/IQR25%/IQR75% for each epoch (~ column)
calculated_values = [data.mean(), data.median(), data.min(), data.max(), data.quantile(0.25), data.quantile(0.75)]
calculated_percentage = [value * 100 for value in calculated_values]
namelist = ['Mean', 'Median', 'Minimum', 'Maximum', 'IQR 25', 'IQR 75%']

# ----------- [Optional] Write results on CVS file ---------- #
# Save the calculated values to the new CSV file
with open('../results/acc_of_each_epoch.csv', 'w', newline='') as epoch_file:
    writer = csv.writer(epoch_file)

    # Write the header row
    writer.writerow(['Epoch'] + [str(k + 1) for k in range(no_of_epoch)])

    # Write the data row by row
    for name, values in zip(namelist, calculated_percentage):
        writer.writerow([name] + [*values])

# ------------------------ Plot results ----------------------- #
options = {
    'title': 'Impact of Number of Epochs on Overall Hyperparameter Sets',
    'xlabel': 'Epoch',
    'xticks': [0, 9, 19, 29, 39, 49, 59, 69, 79],
    'ylabel': 'Accuracy (%)',
}

fig, ax = plt.subplots()

# Plot the trend of the accuracy over the epochs
for name, values in zip(namelist, calculated_percentage):
    ax.plot(values, alpha=0.5, label=f'{name}')

ax.set(**options)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1)
plt.gcf().set_size_inches(12, 7)  # Adjust the figure size
fig.savefig(f"../results/acc_of_each_epoch.png", dpi=300)  # Save the figure as a PNG file
plt.show()

print('Finished')
