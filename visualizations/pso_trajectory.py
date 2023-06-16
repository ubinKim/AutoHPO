"""
PSO trajectory plot representing the positions of the particles and the global best solutions for each iteration.
Principal Component Analysis (PCA) is applied for the dimensionality reduction.

- Create a GIF of the animated PSO trajectory plot (1) using FuncAnimation or (2) from a sequence of images
- x, y-axis: principal component 1 and 2 (PC1, PC2)
- title: the iteration counts
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from PIL import Image


# The function to call at each frame. (i: the current iteration number)
def plot_one_iter(i):
    ax.clear()  # Clear the previous plot

    # updates the position of particles (blue) and gBest (red) in the plot
    p_plot = ax.plot(per_iter[i][:-1, 0], per_iter[i][:-1, 1], 'bo', alpha=0.7, label='particles')
    gBest_plot = ax.plot(per_iter[i][-1, 0], per_iter[i][-1, 1], 'r*', markersize=15, label='gBest')

    # Set the padding values for the background span
    padding = 5
    xmin = min(per_iter[i][:, 0]) - padding
    xmax = max(per_iter[i][:, 0]) + padding

    # Add the colored background span to the scatter plot
    ax.axvspan(xmin, xmax, facecolor='gray', alpha=0.2)

    options = {
        'xlabel': f"PC1 ({explained_variances[0]*100:.2f}%)",
        'xlim': ([-60, 135]),
        'ylabel': f"PC2 ({explained_variances[1]*100:.2f}%)",
        'ylim': ([-0.25, 0.25])
    }

    ax.set(**options)
    title = ax.set_title(f"Iteration {i + 1}")
    title.set_size(17)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.savefig(f"../results/pso_trajectory/iter_{i + 1}.png", dpi=300)

    return ax, p_plot, gBest_plot


# --------------------------- Load data --------------------------- #
particles_file = pd.read_csv('../results/pso_all_particles.csv')
gBest_file = pd.read_csv('../results/pso_gBest.csv')

particles = list(zip(particles_file['lr'], particles_file['dropout'], particles_file['hidden_unit']))
gBest = list(zip(gBest_file['gBest_lr'], gBest_file['gBest_dropout'], gBest_file['gBest_hidden_unit']))

n_particles = 10    # number of particles per iteration
n_iters = 50    # number of iterations

combined = []
for i in range(0, len(particles), n_particles):
    combined.extend(particles[i:i + n_particles])    # 10 particles per iteration
    combined.append(gBest[i // n_particles])         # 1 gBest per iteration

n_particles += 1    # Update the number of particles per iteration

# ------------------------------ PCA ------------------------------ #
# Perform dimensionality reduction using PCA
pca = PCA(n_components=2)  # convert to 2D
combined_2d = pca.fit_transform(combined)

# [PC1, PC2]: the proportion of variance explained by the corresponding principal component in %.
explained_variances = pca.explained_variance_ratio_

# -------------------------- Create plots -------------------------- #
# [Optional] check x and y limits (min/max)
print("xmin:", min(pos[0] for pos in combined_2d))
print("xmax:", max(pos[0] for pos in combined_2d))
print("ymin:", min(pos[1] for pos in combined_2d))
print("ymax:", max(pos[1] for pos in combined_2d))

# Split the list for each iteration
per_iter = [combined_2d[i:i + n_particles] for i in range(0, len(combined_2d), n_particles)]

fig, ax = plt.subplots()
fig.set_tight_layout(True)  # automatically adjusts the layout of the elements within the figure to maximize spacing.

# ------------ Method (1) Create GIF using FuncAnimation ------------ #
'''
FuncAnimation():
    fig: the figure object to animate
    plot_one_iter: the function that updates the animation at each iteration
    frames: the total number of iterations. If an integer, then equivalent to passing range(frames).
    interval: the time interval between frames in milliseconds (1000 ms = 1 s)
    blit: a boolean indicating whether to draw only the parts of the plot that have changed (True) or the entire plot at each iteration (False)
    repeat: a boolean indicating whether to repeat the animation after all the frames have been shown (True) or not (False).
'''
# Make an animation by repeatedly calling a plot_one_iter function.
anim = FuncAnimation(fig, plot_one_iter, frames=n_iters, interval=1000, blit=False, repeat=True)

# Save the animation as a GIF file
writer = PillowWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
anim.save("../results/pso_trajectory_with_FuncAnimation.gif", writer=writer)


# --- Method (2) Create GIF by animating a sequence of saved images --- #
images = []

# Load images named iter_x.png in sequential order (from 1 to n_images)
for i in range(n_iters):
    filename = f'../results/pso_trajectory/iter_{i+1}.png'
    img = Image.open(filename)  # Load image object
    images.append(img)

'''
duration: the time duration (in milliseconds) for each frame
loop=0: an infinite loop for the GIF
'''
# Save the animation as a GIF file
images[0].save('../results/pso_trajectory/pso_trajectory_animated_images.gif',
               save_all=True, append_images=images[1:], duration=200, loop=0)

print("GIF saved successfully!")
