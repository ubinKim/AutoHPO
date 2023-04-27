import itertools
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn import functional

from cls_TIS_dataset import TISDataset
from cls_TIS_model import TISv1

# Optimizers
import random
import pyswarms as ps


def simple_test_run(model, data_loader):
    """
    Args:
        model (Pytorch model): NN model to run
        data_loader (Pytorch DataLoader): Data loader to get predictions on
    """

    # Model is wrapped around DataParallel
    model = model.eval()
    # Output files
    data_size = data_loader.batch_size * len(data_loader)
    correctly_classified = 0

    correct_label_list = []
    pred_label_list = []
    pred_conf_list = []
    pred_logit_list = []
    for data_index, (images, labels, _) in enumerate(data_loader):
        # --- Forward pass begins -- #
        # Convert images and labels to variable
        with torch.no_grad():
            outputs = model(images)
        prob_outputs = functional.softmax(outputs, dim=1)
        # Get predictions
        prediction_confidence, predictions = torch.max(prob_outputs, 1)
        prediction_logit, _ = torch.max(outputs, 1)
        # --- Forward pass ends -- #

        # --- File export output format begins --- #
        correctly_classified += sum(np.where(predictions.numpy() == labels.numpy(), 1, 0))
        # Convert outputs to list
        predicted_as = list(predictions.numpy())
        true_labels = list(labels.numpy())
        prediction_confidence = list(prediction_confidence.numpy())
        prediction_logit = list(prediction_logit.numpy())
        # Extend the big list
        correct_label_list.extend(true_labels)
        pred_label_list.extend(predicted_as)
        pred_conf_list.extend(prediction_confidence)
        pred_logit_list.extend(prediction_logit)
        # --- File export output format ends --- #

    # Calculate accuracy
    acc = "{0:.4f}".format(correctly_classified / data_size)
    return acc, correct_label_list, pred_label_list, pred_conf_list, pred_logit_list


def extract_class_accuracy(correct_labels, pred_labels):
    """
        Get per-class accuracy based on the predictions
    """

    # Initialize the dictionary: [T, F] = [0, 0]
    class_acc = [0 for x in set(correct_labels)]
    class_tot_samples = [0 for x in set(correct_labels)]

    for true, pred in zip(correct_labels, pred_labels):
        class_tot_samples[true] += 1
        if true == pred:
            class_acc[true] += 1

    for index, (class_tot, class_corr) in enumerate(zip(class_tot_samples, class_acc)):
        class_acc[index] = class_acc[index] / class_tot_samples[index]

    return class_acc    # 0 for non-tis, 1 for tis


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


def evaluate(model, train_loader, valid_loader):
    """
    Evaluate the model on the train set and validation set (= test set ).
    """

    # Initialization: [train, valid] = [0, 0]
    acc, tpr, fpr = [[0, 0] for _ in range(3)]

    for index, loader in enumerate([train_loader, valid_loader]):
        acc[index], correct_label_list, pred_label_list, _, _ = simple_test_run(model, loader)
        tpr[index], fpr[index] = extract_class_accuracy(correct_label_list, pred_label_list)

    return acc, tpr, fpr


# Objective function for the single particle (each hyperparameter set)
def objective_func(lr, dropout, hidden_layer):

    # Initialize model and optimizer with current set of hyperparameters
    model = TISv1(dropout=dropout, hidden_layer=hidden_layer)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(no_of_epoch):
        train_one_epoch(model, tr_loader, loss_fn, optimizer)
    acc, _, _ = evaluate(model, tr_loader, val_loader)

    return acc


# Objective function for the whole swarm particles (search_one_iter)
def whole_swarm_obj_func(whole_params):
    whole_val_acc = []
    single_results = {}

    for single in whole_params:
        single_lr = single[0]
        single_drop = single[1]
        single_hidden = int(single[2])

        single_acc = objective_func(single_lr, single_drop, single_hidden)
        single_results[float(single_acc[1])] = [single_lr, single_drop, single_hidden, single_acc[0], single_acc[1]]
        whole_val_acc.append(-float(single_acc[1])) # maximize validation accuracy

    # Save the best particle (best accuracy and corresponding hyperparameter set) of each iteration
    iter_lr, iter_dropout, iter_hidden, iter_tr_acc, iter_val_acc = single_results[max(single_results.keys())]
    iter_best = f"{iter_lr:06.4f},{iter_dropout:06.4f},{iter_hidden:.0f},{iter_tr_acc},{iter_val_acc}\n"
    pso_results_file.write(iter_best)

    return np.array(whole_val_acc)  # val_acc for each particle


if __name__ == "__main__":

    print('Started')

    # Load training and validation datasets
    tr_dataset = TISDataset(['../data/tr_5prime_utr.pos', '../data/tr_5prime_utr.neg'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
    val_dataset = TISDataset(['../data/val_5prime_utr.pos', '../data/val_5prime_utr.neg'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Path to save the results
    header = 'lr,dropout,hidden_layer,tr_acc,val_acc\n'

    # Define number of epochs to train and loss function
    no_of_epoch = 50
    loss_fn = nn.CrossEntropyLoss()

    # --- Grid Search (GS) --- #
    # Define the search space for the grid search (GS)
    grid_lr = [0.1, 0.01, 0.001, 0.0001]  # learning rate
    grid_drop = [0.1, 0.25, 0.5]  # dropout
    grid_hidden = [32, 64, 128, 256]  # number of units in the hidden layer

    # Define the number of points on the search space for both GS and random search (RS).
    num_points = len(grid_lr) * len(grid_drop) * len(grid_hidden)

    # Hyperparameter optimization using GS, RS, and PSO
    with open('../results/grid_results.csv', 'w') as grid_results_file, \
         open('../results/random_results.csv', 'w') as random_results_file, \
         open('../results/pso_results.csv', 'w') as pso_results_file:

        grid_results_file.write(header)
        random_results_file.write(header)
        pso_results_file.write(header)

        # tqdm: print progress bar
        for g_lr, g_drop, g_hidden in tqdm(list(itertools.product(grid_lr, grid_drop, grid_hidden)), total=num_points):

            grid_acc = objective_func(g_lr, g_drop, g_hidden)
            grid_result = f"{g_lr:06.4f},{g_drop:06.4f},{g_hidden},{grid_acc[0]},{grid_acc[1]}\n"
            grid_results_file.writelines(grid_result)

            # --- Random Search (RS) --- #
            # Generate random values for RS
            random_lr = 1 / random.randint(1, 10000)
            random_drop = 1 / random.randint(1, 100)
            random_hidden = random.randint(2, 256)  # must be integer

            random_acc = objective_func(random_lr, random_drop, random_hidden)
            random_result = f"{random_lr:06.4f},{random_drop:06.4f},{random_hidden},{random_acc[0]},{random_acc[1]}\n"
            random_results_file.writelines(random_result)

        # --- Particle Swarm Optimization (PSO) --- #
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Default values
        pso_search_space = (np.array([1e-5, 0, 2]), np.array([1, 1, 256]))  # (lower_bound, upper_bound)

        pso_optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options, bounds=pso_search_space)
        best_val_acc, best_params = pso_optimizer.optimize(whole_swarm_obj_func, iters=50)

        # Save the best particle (best accuracy and corresponding hyperparameter set) of whole iteration
        pso_lr, pso_drop, pso_hidden = best_params
        pso_result = f"{pso_lr:06.4f},{pso_drop:06.4f},{pso_hidden:.0f},,{-best_val_acc},best\n"
        pso_results_file.write(pso_result)

    print('Finished')
