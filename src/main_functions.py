import itertools
from tqdm import tqdm
import time

import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from cls_TIS_dataset import TISDataset
from cls_TIS_model import TISv1

import random
import pyswarms as ps


def calculate_accuracy(model, data_loader):
    """
    Evaluate the performance of the trained NN model on a given dataset in terms of accuracy.

    :param model: Pytorch NN model to run
    :param data_loader: Pytorch Data loader to get predictions on
    :return: acc: accuracy of the trained model on the given dataset
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
    Train the NN model on a training dataset.

    :param model: Pytorch NN model to train
    :param train_loader: Pytorch Data loader to get training data
    :param loss_fn: Pytorch Loss function to calculate loss
    :param optimizer: Pytorch Optimizer to update weights
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

    :param lr: learning rate (numpy.float64)
    :param dropout: probability of dropout (numpy.float64)
    :param hidden_unit: number of units in the hidden layer (int)
    :param tr_loader: Pytorch Data loader to get training data
    :param val_loader: Pytorch Data loader to get validation data
    :param no_of_epoch: number of epochs to train the model

    :return: [tr_acc, val_acc]: the list containing the computed training and validation accuracy
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


def non_optimized(num_models, result_path, obj_args):
    """
    Generate non-optimized model to set the baseline for the comparison.
    Arbitrary hyperparameter configurations are selected, as it was in the random search.
    The only difference is that this function calculate the elapsed_time for each configuration.

    :param num_models: number of non-optimized NN models (to calculate the average)
    :param result_path: the path to the file to save the results

    :return: training and validation accuracy of the model is recorded on the results_file
             for each hyperparameter configuration, with the execution time.
    """

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
            elapsed_time = time.time() - start_time
            non_optimized_result = f"{lr:06.4f},{dropout:06.4f},{hidden_unit},{tr_acc},{val_acc},{elapsed_time}\n"
            non_optimized_file.write(non_optimized_result)

            
def grid_search(search_space, num_config, result_path, obj_args):
    """
    Grid search algorithm for AutoHPO

    :param search_space: search space with the pre-determined values for each hyperparameter
    :param num_config: the number of hyperparameter configurations to be explored
    :param result_path: the path to the file to save the results

    :return: training and validation accuracy of the model is recorded on the results_file
             for each hyperparameter configuration.
    """
    
    start_time = time.time()
    header = 'lr,dropout,hidden_unit,tr_acc,val_acc\n'
    best_hps = [0, 0, 0, 0, 0]  # Initialization

    with open(result_path, 'w') as results_file:
        results_file.write(header)

        for lr, dropout, hidden_unit in tqdm(list(itertools.product(*search_space)), total=num_config):
            # calculate the accuracy after training the model for the given number of epochs
            tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
            current_hps = [lr, dropout, hidden_unit, tr_acc, val_acc]
            current_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{}\n".format(*current_hps)
            results_file.write(current_hps_record)
            
            # update best_hps
            if best_hps[4] < current_hps[4]:
                best_hps = current_hps

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for GS: {elapsed_time:.2f} seconds")

        # Save best hyperparameter set with the total elapsed time
        best_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{},best HPs with {:.2f} s\n".format(*best_hps, elapsed_time)
        results_file.write(best_hps_record)


def grid_with_epoch(search_space, num_config, result_path, epoch_path, tr_loader, val_loader, no_of_epoch):
    """
    Grid search algorithm for AutoHPO

    In addition to the final results after training the model for the given number of epochs,
    the validation accuracy of each hyperparameter configurations over the epochs is recorded.

    :param search_space: search space with the pre-determined values for each hyperparameter
    :param num_config: the number of hyperparameter configurations to be explored
    :param result_path: the path to the file to save the results after training the model for the given number of epochs
    :param epoch_path: the path to the file to save the validation accuracy of each hyperparameter configurations
           over the epochs
    :param tr_loader: Pytorch Data loader to get training data
    :param val_loader: Pytorch Data loader to get validation data
    :param no_of_epoch: number of epochs to train the model

    :return: training and validation accuracy of the model is recorded on the results_file and epoch_file
             for each hyperparameter configuration.
    """

    start_time = time.time()
    header = 'lr,dropout,hidden_unit,tr_acc,val_acc\n'
    best_hps = [0, 0, 0, 0, 0]  # Initialization
    loss_fn = nn.CrossEntropyLoss()

    with open(result_path, 'w', newline='') as results_file, \
         open(epoch_path, 'w', newline='') as epoch_file:

        # Write the header row
        results_file.write(header)
        epoch_writer = csv.writer(epoch_file)
        epoch_writer.writerow(['Epoch'] + [str(k + 1) for k in range(no_of_epoch)])

        for lr, dropout, hidden_unit in tqdm(list(itertools.product(*search_space)), total=num_config):

            # Initialize model and optimizer with a given set of hyperparameters
            model = TISv1(dropout=dropout, hidden_unit=hidden_unit)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            acc_per_hps = []

            for epoch in range(no_of_epoch):
                train_one_epoch(model, tr_loader, loss_fn, optimizer)
                val_acc = calculate_accuracy(model, val_loader)
                acc_per_hps.append(val_acc)

            # row: all val_acc of each HP set over the epochs
            epoch_writer.writerow([f'HP({lr}, {dropout}, {hidden_unit})'] + acc_per_hps)

            # calculate the training accuracy only after training the model for the given number of epochs
            tr_acc = calculate_accuracy(model, tr_loader)
            current_hps = [lr, dropout, hidden_unit, tr_acc, val_acc]
            current_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{}\n".format(*current_hps)
            results_file.write(current_hps_record)

            # update best_hps
            if best_hps[4] < current_hps[4]:
                best_hps = current_hps

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for GS: {elapsed_time:.2f} seconds")

        # Save best hyperparameter configuration with the total elapsed time
        best_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{},best HPs with {:.2f} s\n".format(*best_hps, elapsed_time)
        results_file.write(best_hps_record)
        
        
def random_search(num_config, result_path, obj_args):
    """
    Random search algorithm for AutoHPO

    :param num_config: the number of hyperparameter configurations to be explored
    :param result_path: the path to the file to save the results

    :return: training and validation accuracy of the model is recorded on the results_file
             for each hyperparameter configuration.
    """

    start_time = time.time()
    header = 'lr,dropout,hidden_unit,tr_acc,val_acc\n'
    best_hps = [0, 0, 0, 0, 0]  # Initialization

    with open(result_path, 'w') as results_file:
        results_file.write(header)

        for _ in tqdm(range(num_config)):

            # Random initialization for RS
            lr = 1 / random.randint(10, 10000)
            dropout = 1 / random.randint(2, 10)
            hidden_unit = random.randint(32, 256)  # must be integer

            tr_acc, val_acc = objective(lr, dropout, hidden_unit, **obj_args)
            current_hps = [lr, dropout, hidden_unit, tr_acc, val_acc]
            current_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{}\n".format(*current_hps)
            results_file.write(current_hps_record)

            # update best_hps
            if best_hps[4] < current_hps[4]:
                best_hps = current_hps

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for RS: {elapsed_time:.2f} seconds")

    # Save best hyperparameter configuration with the total elapsed time
    best_hps_record = "{:06.4f},{:06.4f},{:.0f},{},{},best HPs with {:.2f} s\n".format(*best_hps, elapsed_time)
    results_file.write(best_hps_record)
    
            
def search_one_iter(swarm, gBest_file, particles_file, obj_args, update_args):
    """
    Apply the objective function to the swarm of particles.

    :param swarm: the position of each particle in the swarm | numpy.ndarray of shape (n_particles, dimensions)
    :param gBest_file: the file to save the global best position and the corresponding accuracies
    :param particles_file: the file to save the position and the corresponding accuracies of each PSO particle
    :param update_args: arguments to update the iteration counts and the global best particle explored so far

    :return: swarm_val_acc: the set of computed validation accuracy (with a negative sign) for each particle
             | numpy.ndarray of shape (n_particles, )
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
    """
    Particle Swarm Optimization algorithm for AutoHPO

    :param gBest_path: the path to the file to save the global best position and the corresponding accuracies
    :param particles_path: the path to file to save the position and the corresponding accuracies of each PSO particle
    :param options: default values for c1, c2, and w
    :param bounds: the range of search space for each hyperparameter (lower_bound, upper_bound)
    :param n_particles: the number of PSO particles in the swarm
    :param iters: the maximum number of iterations (termination condition)

    :return: the data related to the global best particle is recorded on the gBest_file
             while the data related to the each PSO particle is recorded on the particles_file over the iterations
    """
    
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
        pso_optimizer.optimize(search_one_iter, iters=iters, **iter_args)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time for PSO: {elapsed_time:.2f} seconds")
