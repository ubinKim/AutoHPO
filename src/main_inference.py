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

if __name__ == "__main__":

    print('Started')

    # Load training and validation datasets
    tr_dataset = TISDataset(['../data/tr_5prime_utr.pos', '../data/tr_5prime_utr.neg'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
    val_dataset = TISDataset(['../data/val_5prime_utr.pos', '../data/val_5prime_utr.neg'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Path to save the model and results
    header = 'lr,dropout,hidden_layer,tr_acc,val_acc\n'
    grid_model_path = '../results/tis_v1_grid.pth'
    grid_results_file = open('../results/grid_results.csv', 'w')
    grid_results_file.write(header)
    random_model_path = '../results/tis_v1_random.pth'
    random_results_file = open('../results/random_results.csv', 'w')
    random_results_file.write(header)

    # Number of epochs to train
    no_of_epoch = 50

    # Define a loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the search space for the grid search
    grid_lr = [0.1, 0.01, 0.001, 0.0001]    # learning rate
    grid_drop = [0.1, 0.25, 0.5]    # dropout
    grid_hidden = [32, 64, 128, 256]  # number of units in the hidden layer

    # Hyperparameter optimization using grid search and random search
    for lr in grid_lr:
        for dropout in grid_drop:
            for hidden_layer in grid_hidden:

                # Generate random values for the random search
                random_lr = 1 / random.randint(1, 1000)
                random_drop = 1 / random.randint(1, 100)
                random_hidden = random.randint(2, 256)  # must be integer

                # Initialize model and optimizer with current set of hyperparameters
                grid_model = TISv1(dropout=dropout, hidden_layer=hidden_layer)
                grid_optimizer = torch.optim.SGD(grid_model.parameters(), lr=lr)
                random_model = TISv1(dropout=random_drop, hidden_layer=random_hidden)
                random_optimizer = torch.optim.SGD(random_model.parameters(), lr=random_lr)

                # Training
                for epoch in range(no_of_epoch):
                    train_one_epoch(grid_model, tr_loader, loss_fn, grid_optimizer)
                    train_one_epoch(random_model, tr_loader, loss_fn, random_optimizer)

                # Evaluation
                grid_acc, grid_tpr, grid_fpr = evaluate(grid_model, tr_loader, val_loader)
                random_acc, random_tpr, random_fpr = evaluate(random_model, tr_loader, val_loader)

                # Save the model and results
                grid_result = f"{lr:06.4f},{dropout:06.4f},{hidden_layer},{grid_acc[0]},{grid_acc[1]}\n"
                grid_results_file.writelines(grid_result)
                torch.save(grid_model.state_dict(), grid_model_path)
                random_result = f"{random_lr:06.4f},{random_drop:06.4f},{random_hidden},{random_acc[0]},{random_acc[1]}\n"
                random_results_file.writelines(random_result)
                torch.save(random_model.state_dict(), random_model_path)

    grid_results_file.close()
    random_results_file.close()
    print('Finished')
