
from main_functions import *

print('Started')

# Load training and validation datasets
tr_dataset = TISDataset(['../data/tr_5prime_utr.pos', '../data/tr_5prime_utr.neg'])
tr_loader = DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
val_dataset = TISDataset(['../data/val_5prime_utr.pos', '../data/val_5prime_utr.neg'])
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# Define the arguments as a dictionary
obj_args = {
    'tr_loader': tr_loader,
    'val_loader': val_loader,
    'no_of_epoch': 80
}

non_optimized_args = {
    'num_models': 10,    # Number of non-optimized NN models (to calculate the average)
    'result_path': '../results/non_optimized.csv',
    'obj_args': obj_args,
}

grid_args = {
    'search_space': [[0.0001, 0.001, 0.01, 0.1], [0.1, 0.17, 0.29, 0.5], [32, 64, 128, 256]],
    'num_config': 64,    # 4*4*4
    'result_path': '../results/grid_results.csv',
    'obj_args': obj_args,
}

grid_with_epoch_args = {
    'search_space': [[0.0001, 0.001, 0.01, 0.1], [0.1, 0.17, 0.29, 0.5], [32, 64, 128, 256]],
    'num_config': 64,  # 4*4*4
    'result_path': '../results/grid_results.csv',
    'epoch_path': '../results/grid_with_epoch_results.csv',
    'tr_loader': tr_loader,
    'val_loader': val_loader,
    'no_of_epoch': 80
}

random_args = {
    'num_config': 64,    # 4*4*4
    'result_path': '../results/random_results.csv',
    'obj_args': obj_args,
}

pso_args = {
    'gBest_path': '../results/pso_gBest.csv',
    'particles_path': '../results/pso_all_particles.csv',
    'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9},    # Default values
    'bounds': (np.array([0.0001, 0.1, 32]), np.array([0.1, 0.5, 256])),    # (lower_bound, upper_bound)
    'n_particles': 10,
    'iters': 50,    # total number of iterations
    'obj_args': obj_args
}

# run the functions
non_optimized(**non_optimized_args)
grid_search(**grid_args)
grid_with_epoch(**grid_with_epoch_args)
random_search(**random_args)
pso(**pso_args)

print('Finished')
