import pdb
import numpy as np
from sklearn.metrics import roc_curve
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from cls_TIS_dataset import TISDataset
from cls_TIS_model import TISv1
import seaborn as sns
import matplotlib.pyplot as plt

def simple_test_run(model, data_loader):
    """
    Args:
        model (Pytorch model): NN model to run
        data_loader (Pytorch DataLoader): Data loader to get predictions on
        file_path (string): Path to file that contains exported results
        file_name (string): File  name to  export results
        run_mode (string): Name of the run, for printing etc
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
        # images = images
        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        prob_outputs = functional.softmax(outputs, dim=1)
        # Get predictions
        prediction_confidence, predictions = torch.max(prob_outputs, 1)
        prediction_logit, _ = torch.max(outputs, 1)
        # --- Forward pass ends -- #

        # --- File export output format begins --- #
        correctly_classified += sum(np.where(predictions.numpy() == labels.numpy(), 1, 0))
        # Convert outs to list
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


# def load_pytorch_model(folder_path, file_name):
#     """
#     Args:
#         folder_path (string): Path to file that results will be read from
#         file_name (string): Name of the file that results will be read from
#
#     returns:
#         model (Pytorch model): loaded model
#     """
#     file_with_path = os.path.join(folder_path, file_name)
#     model = torch.load(file_with_path)
#     return model


def calc_fpr_at_tpr80(true_labels, pred_labels, pred_conf):
    y_score = []
    for pred, conf in zip(pred_labels, pred_conf):
        if pred == 0:
            conf = 1 - conf
        y_score.append(conf)

    fpr, tpr, thresholds = roc_curve(true_labels, y_score, pos_label=1)

    # Calculate the FPR at TPR >= 80 %
    for f, t in zip(fpr, tpr):
        if t >= 0.8:
            return f
    return 0


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    """
    Args:
        model (Pytorch model): NN model to train
        train_loader (Pytorch DataLoader): Data loader to get training data
        valid_loader (Pytorch DataLoader): Data loader to get validation data
        loss_fn (Pytorch loss function): Loss function to calculate loss
        optimizer (Pytorch optimizer): Optimizer to update weights
        n_epochs (int): Number of epochs to train
        model_path (string): Path to file that contains exported model
    """
    train_loss = 0.0
    valid_loss = 0.0
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
    # print(f"Epoch: {epoch+1}/{n_epochs}\tTraining Loss: {train_loss:.4f}\tValidation Loss: {valid_loss:.4f}")

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
    tr_dataset = TISDataset(['tr_5prime_utr_pos', 'tr_5prime_utr_neg'])
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=64, shuffle=True)
    val_dataset = TISDataset(['val_5prime_utr.pos', 'val_5prime_utr.neg'])
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    # Path to save the model and results
    header = 'lr,dropout,hidden_layer,tr_acc,val_acc\n'
    grid_model_path = 'tis_v1_grid.pth'
    grid_results_file = open('grid_results.csv', 'w')
    grid_results_file.write(header)

    # Number of epochs to train
    no_of_epoch = 80

    # Define a loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the search space for the grid search
    grid_lr = [0.1, 0.01, 0.001, 0.0001]    # learning rate
    grid_drop = [0.1, 0.25, 0.5]    # dropout
    grid_hidden = [32, 64, 128, 256]  # number of units in the hidden layer

    # Hyperparameter optimization using grid search and random search
    best_tr_acc = 0
    best_val_acc = 0
    final_lr = 0
    final_dropout = 0
    final_hidden_layer = 0
    valid_acc = 0

    tr_accuracy_matrix_ld = np.zeros((len(grid_lr), len(grid_drop)))
    tr_accuracy_matrix_lh = np.zeros((len(grid_lr), len(grid_hidden)))
    tr_accuracy_matrix_dh = np.zeros((len(grid_drop), len(grid_hidden)))

    val_accuracy_matrix_ld = np.zeros((len(grid_lr), len(grid_drop)))
    val_accuracy_matrix_lh = np.zeros((len(grid_lr), len(grid_hidden)))
    val_accuracy_matrix_dh = np.zeros((len(grid_drop), len(grid_hidden)))

    for i, lr in enumerate(grid_lr):
        for j, dropout in enumerate(grid_drop):
            for k, hidden_layer in enumerate(grid_hidden):

                # Initialize model and optimizer with current set of hyperparameters
                grid_model = TISv1(dropout=dropout, hidden_layer=hidden_layer)
                grid_optimizer = torch.optim.SGD(grid_model.parameters(), lr=lr)

                # Training
                for epoch in range(no_of_epoch):
                    train_one_epoch(grid_model, tr_loader, loss_fn, grid_optimizer)

                # Evaluation
                grid_acc, grid_tpr, grid_fpr = evaluate(grid_model, tr_loader, val_loader)

                # Making train matrix
                if k == 0:
                    tr_accuracy_matrix_ld[i][j] = float(grid_acc[0])
                # if tr_accuracy_matrix_ld[i][j] has higher accuracy with higher k, updates tr_accuracy_matrix_ld[i][j]
                else:
                    if tr_accuracy_matrix_ld[i][j] < float(grid_acc[0]):
                        tr_accuracy_matrix_ld[i][j] = float(grid_acc[0])
                    else:
                        pass
                if j == 0:
                    tr_accuracy_matrix_lh[i][k] = float(grid_acc[0])
                # if tr_accuracy_matrix_lh[i][k] has higher accuracy with higher j, updates tr_accuracy_matrix_lh[i][k]
                else:
                    if tr_accuracy_matrix_lh[i][k] < float(grid_acc[0]):
                        tr_accuracy_matrix_lh[i][k] = float(grid_acc[0])
                    else:
                        pass
                if i == 0:
                    tr_accuracy_matrix_dh[j][k] = float(grid_acc[0])
                # if tr_accuracy_matrix_dh[j][k] has higher accuracy with higher i, updates tr_accuracy_matrix_dh[j][k]
                else:
                    if tr_accuracy_matrix_dh[j][k] < float(grid_acc[0]):
                        tr_accuracy_matrix_dh[j][k] = float(grid_acc[0])
                    else:
                        pass

                # Making valid matrix
                if k == 0:
                    val_accuracy_matrix_ld[i][j] = float(grid_acc[1])
                # if val_accuracy_matrix_ld[i][j] has higher accuracy with higher k, updates val_accuracy_matrix_ld[i][j]
                else:
                    if val_accuracy_matrix_ld[i][j] < float(grid_acc[1]):
                        val_accuracy_matrix_ld[i][j] = float(grid_acc[1])
                    else:
                        pass
                if j == 0:
                    val_accuracy_matrix_lh[i][k] = float(grid_acc[1])
                # if val_accuracy_matrix_lh[i][k] has higher accuracy with higher j, updates val_accuracy_matrix_lh[i][k]
                else:
                    if val_accuracy_matrix_lh[i][k] < float(grid_acc[1]):
                        val_accuracy_matrix_lh[i][k] = float(grid_acc[1])
                    else:
                        pass
                if i == 0:
                    val_accuracy_matrix_dh[j][k] = float(grid_acc[1])
                # if val_accuracy_matrix_dh[j][k] has higher accuracy with higher i, updates val_accuracy_matrix_dh[j][k]
                else:
                    if val_accuracy_matrix_dh[j][k] < float(grid_acc[1]):
                        val_accuracy_matrix_dh[j][k] = float(grid_acc[1])
                    else:
                        pass

                # Save all the combinations of 3 hyperparameter in csv file
                grid_result = f"{lr:06.4f},{dropout:06.4f},{hidden_layer},{grid_acc[0]},{grid_acc[1]}\n"
                # Save the model
                # torch.save(model.state_dict(), model_path)
                grid_results_file.writelines(grid_result)
                torch.save(grid_model.state_dict(), grid_model_path)

                # to identify the best train accuracy with 3 hyperparameter combination and valid accuracy of it
                if float(grid_acc[0]) > best_tr_acc:
                    final_lr = lr
                    final_dropout = dropout
                    final_hidden_layer = hidden_layer
                    best_tr_acc = float(grid_acc[0])
                    valid_acc = grid_acc[1]
    # results_file.close()
    grid_results_file.close()

    # print out (Best train accuracy, its 3 hyperparameter, and its valid(test) accuracy)
    best_grid_result = f"{final_lr:06.4f},{final_dropout:06.4f},{final_hidden_layer},{best_tr_acc},{valid_acc}"
    print(best_grid_result)


    plt.subplot(231)
    tr_map_ld = sns.heatmap(tr_accuracy_matrix_ld, annot=True, fmt=".4g", xticklabels=grid_drop,
                          yticklabels=grid_lr, cmap='summer')
    tr_map_ld.set_title("Tr Accuracy Heatmap")
    tr_map_ld.set_xlabel("Hidden Size")
    tr_map_ld.set_ylabel("Dropout")

    plt.subplot(232)
    tr_map_lh = sns.heatmap(tr_accuracy_matrix_lh, annot=True, fmt=".4g", xticklabels=grid_hidden,
                          yticklabels=grid_lr, cmap='summer')
    tr_map_lh.set_title("Tr Accuracy Heatmap")
    tr_map_lh.set_xlabel("Hidden Size",)
    tr_map_lh.set_ylabel("Learning Rate")

    plt.subplot(233)
    tr_map_dh = sns.heatmap(tr_accuracy_matrix_dh, annot=True, fmt=".4g", xticklabels=grid_hidden,
                          yticklabels=grid_drop, cmap='summer')
    tr_map_dh.set_title("Tr Accuracy Heatmap")
    tr_map_dh.set_xlabel("Dropout")
    tr_map_dh.set_ylabel("Hidden Size")

    plt.subplot(234)
    val_map_ld = sns.heatmap(val_accuracy_matrix_ld, annot=True, fmt=".4g", xticklabels=grid_drop,
                                  yticklabels=grid_lr, cmap='summer')
    val_map_ld.set_title("Val Accuracy Heatmap")
    val_map_ld.set_xlabel("Dropout")
    val_map_ld.set_ylabel("Learning Rate")

    plt.subplot(235)
    val_map_lh = sns.heatmap(val_accuracy_matrix_lh, annot=True, fmt=".4g", xticklabels=grid_hidden,
                                  yticklabels=grid_lr, cmap='summer')
    val_map_lh.set_title("Val Accuracy Heatmap")
    val_map_lh.set_xlabel("Hidden Size")
    val_map_lh.set_ylabel("Learning Rate")

    plt.subplot(236)
    val_map_dh = sns.heatmap(val_accuracy_matrix_dh, annot=True, fmt=".4g", xticklabels=grid_hidden,
                                  yticklabels=grid_drop, cmap='summer')
    val_map_dh.set_title("Val Accuracy Heatmap")
    val_map_dh.set_xlabel("Hidden Size")
    val_map_dh.set_ylabel("Dropout")

    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    plt.show()
    print('Finished')