"""
Progress of cost function values throughout optimization depending on the number of PSO particles used.

- x-axis: the number of iterations
- y-axis: validation accuracy of gBest
- legend: the number of swarm particles (n_particles = 5, 10, 15)
"""

import matplotlib.pyplot as plt
import csv

with open('../results/pso_80_p5.csv', "r") as p5_file,\
     open('../results/pso_80_p10.csv', "r") as p10_file,\
     open('../results/pso_80_p15.csv', "r") as p15_file:

    fig, ax = plt.subplots()

    for i, pfile in enumerate([p5_file, p10_file, p15_file]):
        p_reader = csv.reader(pfile)
        next(p_reader)  # skip the header row
        p_cost_history = [float(row[-1]) * 100 for row in p_reader]  # Convert to percentage
        ax.plot(p_cost_history, label=f'{(i + 1) * 5} Particles')

    options = {
        'title': 'Progress of Accuracy Depending on the Number of PSO Particles',
        'xlabel': 'Iteration',
        'ylabel': 'Accuracy (%)',
    }
    ax.set(**options)
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"../results/pso_cost.png", dpi=300)  # Save the figure as a PNG file
    plt.show()

print('Finished')
