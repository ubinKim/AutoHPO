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
#import pyswarms as ps


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


def train_one_epoch(model, train_loader, loss_fn):
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

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()

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
    tr_dataset = TISDataset(['tr_5prime_utr.pos', 'tr_5prime_utr.neg'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
    val_dataset = TISDataset(['val_5prime_utr.pos', 'val_5prime_utr.neg'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    no_of_epoch = 1
    loss_fn = nn.CrossEntropyLoss()
    # Initialize model and optimizer with current set of hyperparameters
    nonopti_model = TISv1(dropout=0.5, hidden_layer=256)
    #random_optimizer = torch.optim.SGD(random_model.parameters(), lr=0.001)

    # Training
    for epoch in range(no_of_epoch):
        train_one_epoch(nonopti_model, tr_loader, loss_fn)

    # Evaluation
    nonopti_acc, nonopti_tpr, nonopti_fpr = evaluate(nonopti_model, tr_loader, val_loader)


    print('nonopti_accuracy:{}'.format(nonopti_acc))
    print('nonopti_tpr: {}'.format(nonopti_tpr))
    print('nonopti_fpr: {}'.format(nonopti_fpr))
    print('Finished')
