"""
Create a simple line plot with three lines, each representing the cost history of a PSO algorithm 
with a different number of swarm particles.

Load the cost history data from CSV file. (file name: pso_#epoch_#particles_iter-step-size.csv)

Returns:
-------
matplotlib.axes._subplots.AxesSubplot
    The plot of the cost history throughout optimization.

    x-axis: the number of iterations \n
    y-axis: validation accuracy of gBest \n
    legend: the number of swarm particles (n_particles = 5, 10, 15) \n

"""

import matplotlib.pyplot as plt
import csv

with open('../results/pso_80_5_1.csv', "r") as p5_file,\
     open('../results/pso_80_10_1.csv', "r") as p10_file,\
     open('../results/pso_80_15_1.csv', "r") as p15_file:

    fig, ax = plt.subplots()

    for i, pfile in enumerate([p5_file, p10_file, p15_file]):
        p_reader = csv.reader(pfile)
        next(p_reader)  # skip the header row
        p_cost_history = [float(row[-1]) for row in p_reader]
        ax.plot(p_cost_history, label=f'n_particle: {(i+1)*5}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Accuracy')
    plt.title('Cost History: Accuracy over Iterations')
    ax.legend()
    plt.show()
