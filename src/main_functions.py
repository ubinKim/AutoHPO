import itertools
from tqdm import tqdm
import time

import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional
from cls_TIS_model import TISv1

import random
import pyswarms as ps


def calculate_accuracy(model, data_loader):
    """
    Evaluate the performance of the trained NN model on a given dataset in terms of accuracy.

    Args:
        model (Pytorch model): NN model to run
        data_loader (Pytorch DataLoader): Data loader to get predictions on
    """

    model = model.eval()
    data_size = data_loader.batch_size * len(data_loader)
    correctly_classified = 0

    for data_index, (images, labels, _) in enumerate(data_loader):
        # --- Forward pass begins -- #
        # Convert images and labels to variable
        with torch.no_grad():
            outputs = model(images)
        prob_outputs = functional.softmax(outputs, dim=1)
        # Get predictions
        _, predictions = torch.max(prob_outputs, 1)
        # --- Forward pass ends -- #

        correctly_classified += sum(np.where(predictions.numpy() == labels.numpy(), 1, 0))

    # Calculate accuracy
    acc = "{0:.4f}".format(correctly_classified / data_size)
    return float(acc)


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    """
    Args:
        model (Pytorch model): NN model to train
        train_loader (Pytorch DataLoader): Data loader to get training data
        loss_fn (Pytorch loss function): Loss function to calculate loss
        optimizer (Pytorch optimizer): Optimizer to update weights
    """

    train_loss = 0.0

    # Training loop
    model.train()
    for i, (images, labels, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Average loss
    train_loss /= len(train_loader)


def objective(lr, dropout, hidden_unit, tr_loader, val_loader, no_of_epoch):
    """
    Objective function that computes the training and validation accuracy of the model
    trained with a given set of hyperparameters (single particle in PSO)

    Args:
        lr: learning rate (numpy.float64)
        dropout: probability of dropout (numpy.float64)
        hidden_unit: number of units in the hidden layer (int)

    Returns:
        [train_acc, val_acc]: the list containing the computed training and validation accuracy
    """

    # Initialize model and optimizer with a given set of hyperparameters
    model = TISv1(dropout=dropout, hidden_unit=hidden_unit)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(no_of_epoch):
        train_one_epoch(model, tr_loader, loss_fn, optimizer)

    tr_acc = calculate_accuracy(model, tr_loader)
    val_acc = calculate_accuracy(model, val_loader)

    return [tr_acc, val_acc]


def grid_search(search_space, num_points, result_path, obj_args):

    start_time = time.time()
    header = 'lr,dropout,hidden_unit,tr_acc,val_acc\n'
    best_hps = [0, 0, 0, 0, 0]  # Initialization

    # grid search space
    grid_lr = search_space[0]
    grid_drop = search_space[1]
    grid_hidden = search_space[2]

    with open(result_path, 'w') as results_file:
        results_file.write(header)

        for lr, dropout, hidden_unit in tqdm(list(itertools.product(grid_lr, grid_drop, grid_hidden)), total=num_points):
            tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
            current_hps = [lr, dropout, hidden_unit, tr_acc, val_acc]
            current_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{}\n".format(*current_hps)
            results_file.write(current_hps_record)
            
            # update best_hps
            if best_hps[4] < current_hps[4]:
                best_hps = current_hps

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for GS: {elapsed_time:.2f} seconds")

        # Save best hyperparameter set with the total elapsed time
        best_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{},{}\n".format(*best_hps, elapsed_time)
        results_file.write(best_hps_record)


def random_search(num_points, result_path, obj_args):

    start_time = time.time()
    header = 'lr,dropout,hidden_unit,tr_acc,val_acc\n'

    with open(result_path, 'w') as results_file:
        results_file.write(header)

        for _ in tqdm(range(num_points)):

            # Random initialization for RS
            lr = 1 / random.randint(10, 10000)
            dropout = 1 / random.randint(2, 10)
            hidden_unit = random.randint(32, 256)  # must be integer

            tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
            random_result = f"{lr:06.4f},{dropout:06.4f},{hidden_unit},{tr_acc},{val_acc}\n"
            results_file.write(random_result)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for RS: {elapsed_time:.2f} seconds")


def non_optimized(num_models, result_path, obj_args):

    header = 'lr,dropout,hidden_unit,tr_acc,val_acc,elapsed_time\n'

    with open(result_path, 'w') as non_optimized_file:
        non_optimized_file.write(header)

        for _ in tqdm(range(num_models)):
            start_time = time.time()

            # Random initialization to setting hyperparameters
            lr = 1 / random.randint(10, 10000)
            dropout = 1 / random.randint(2, 10)
            hidden_unit = random.randint(32, 256)  # must be integer

            tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
            end_time = time.time()
            elapsed_time = end_time - start_time
            non_optimized_result = f"{lr:06.4f},{dropout:06.4f},{hidden_unit},{tr_acc},{val_acc},{elapsed_time}\n"
            non_optimized_file.write(non_optimized_result)

            
def search_one_iter(swarm, gBest_file, particles_file, obj_args, update_args):
    """
    Apply the objective function to the swarm of particles.

    Args:
        swarm: the position of each particle in the swarm | numpy.ndarray of shape (n_particles, dimensions)

    Returns:
        swarm_val_acc: the set of computed validation accuracy (with a negative sign) for each particle | numpy.ndarray of shape (n_particles, )
    """

    update_args['iter_count'] += 1  # Increment the iteration count in update_args

    swarm_data = []

    for particle in swarm:
        lr = particle[0]
        dropout = particle[1]
        hidden_unit = int(particle[2])
        tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
        swarm_data.append([lr, dropout, hidden_unit, tr_acc, val_acc])

        # Save all particles of this iteration
        particles_record = "{},{:06.4f},{:06.4f},{:.0f},{},{}\n".format(update_args['iter_count'], *swarm_data[-1])
        particles_file.write(particles_record)

    # Save the best particle of this iteration with the highest validation accuracy
    iter_best = max(swarm_data, key=lambda x: x[4])

    if update_args['gBest'][4] < iter_best[4]:
        update_args['gBest'] = iter_best   # update gBest in update_args

    gBest_record = "{},{:06.4f},{:06.4f},{:.0f},{},{}\n".format(update_args['iter_count'], *update_args['gBest'])
    gBest_file.write(gBest_record)

    # By default, PSO optimizer minimizes the objective function.
    # Add a negative sign to maximize the validation accuracy.
    swarm_val_acc = [-float(particle_data[4]) for particle_data in swarm_data]

    return np.array(swarm_val_acc)


def pso(gBest_path, particles_path, options, bounds, n_particles, iters, obj_args):

    start_time = time.time()

    gBest_header = 'n_iter,gBest_lr,gBest_dropout,gBest_hidden_unit,gBest_tr_acc,gBest_val_acc\n'
    particles_header = 'n_iter,lr,dropout,hidden_unit,tr_acc,val_acc,\n'

    with open(gBest_path, 'w') as gBest_file,\
         open(particles_path, 'w') as particles_file:

        update_args = {
            'iter_count': 0,  # initiation
            'gBest': [0, 0, 0, 0, 0],  # initiation
        }

        iter_args = {
            'gBest_file': gBest_file,
            'particles_file': particles_file,
            'obj_args': obj_args,
            'update_args': update_args
        }

        gBest_file.write(gBest_header)
        particles_file.write(particles_header)
        pso_optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=3, options=options, bounds=bounds)
        best_val_acc, best_params = pso_optimizer.optimize(search_one_iter, iters=iters, **iter_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for PSO: {elapsed_time:.2f} seconds")
