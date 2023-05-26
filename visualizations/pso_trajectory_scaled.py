import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --------------------------- Load data --------------------------- #
gBest_file = pd.read_csv('../results/pso_gBest_epoch80.csv')
particles_file = pd.read_csv('../results/pso_all_particles_epoch80.csv')

particles = list(zip(particles_file['lr'], particles_file['dropout'], particles_file['hidden_unit']))
gBest = list(zip(gBest_file['gBest_lr'], gBest_file['gBest_dropout'], gBest_file['gBest_hidden_unit']))

# ----------------------- Data preprocessing ----------------------- #
# Create a combined list by adding gBest as the n+1 th particle to each iteration
combined = []
n_particles = 10    # the number of particles per iter

for i in range(0, len(particles), n_particles):
    combined.extend(particles[i:i + n_particles])    # 10 particles per iter
    combined.append(gBest[i // n_particles])         # 1 gBest per iter

n_particles += 1    # Update the number of particles per iter

# Data scaling (mean = 0, variance = 1)
scaler = StandardScaler()
combined_and_scaled = scaler.fit_transform(combined)

# Perform dimensionality reduction using PCA on the whole position vector
pca = PCA(n_components=2)  # convert to 2D
combined_2d = pca.fit_transform(combined_and_scaled)

# [PC1, PC2]: the proportion of variance explained by the corresponding principal component in %.
explained_variances = pca.explained_variance_ratio_

# [Optional] check x and y limits (min/max)
print("xmin:", min(pos[0] for pos in combined_2d))
print("xmax:", max(pos[0] for pos in combined_2d))
print("ymin:", min(pos[1] for pos in combined_2d))
print("ymax:", max(pos[1] for pos in combined_2d))

# -------------------------- Create plots -------------------------- #
for i in range(0, len(combined_2d), n_particles):
    # Split the list for each iteration
    per_iter = combined_2d[i:i + n_particles]

    # Plot the particles from this iteration in blue and gBest in red
    fig, ax = plt.subplots()    # Create a new figure for each iteration
    ax.plot(per_iter[-1, 0], per_iter[-1, 1], 'r*', markersize=15, label='gBest')
    ax.plot(per_iter[:-1, 0], per_iter[:-1, 1], 'bo', alpha=0.7, label='particles')

    options = {
        'title': f"Iteration {(i // n_particles) + 1}",
        'xlabel': f"PC1 ({explained_variances[0]*100:.2f}%)",
        'xlim': [-3, 7],
        'ylabel': f"PC2 ({explained_variances[1]*100:.2f}%)",
        'ylim': [-2.5, 2.5]
    }
    ax.set(**options)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.set_tight_layout(True)
    fig.savefig(f"../results/trajectory_images_scaled/iter_{(i // n_particles) + 1}.png", dpi=300)   # Save the figure as a PNG file
    plt.close(fig)  # Close the figure to release memory

print("Images saved successfully!")
