import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.NetworkModels import MotorLearningRNN
from utils.DataGen import generate_synthetic_data
from utils.TorchFunctions import train_evaluate_model, vulnerability_test
from joblib import Parallel, delayed

lr = 0.02

def run_model():
    # Hyperparameters
    input_size = 7  # Number of features in the input. For motor learning, this could represent different aspects of the task or context.
    hidden_size = 7  # Number of features in the hidden state
    output_size = 4  # For simplicity, let's say we're predicting 4 different outcomes related to motor learning

    batch_size = 20

    num_training_sequences = 3  # Number of different training sequences (e.g., different motor tasks)
    num_pre_training_sequences = 10  # Number of different pre-training sequences (e.g., simpler motor tasks)
    num_test_sequences = 100  # Number of different test sequences (e.g., novel motor tasks)

    # lr
    global lr
    print('Learning Rate:', lr)

    # Generate synthetic data for both practices
    X_blocked, y_blocked, X_random, y_random = generate_synthetic_data(num_training_sequences, num_samples=5000, input_noise=True)
    print(X_blocked.shape)
    print(y_blocked.shape)

    X_pre, y_pre, X_pre_random, y_pre_random = generate_synthetic_data(num_pre_training_sequences, num_samples=100, input_noise=True)
    _, _, X_test, y_test = generate_synthetic_data(num_test_sequences, input_noise=True)

    # Instantiate the model
    model1 = MotorLearningRNN(input_size, hidden_size, output_size, num_outputs=input_size)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.0)

    ### Pre-training and pre-test
    total_num_epochs_pre = X_pre.shape[0] // batch_size
    print(f"Total number of epochs for pre-training: {total_num_epochs_pre}")
    _, _, loss_test_pre, _, loss_test_array_pre = train_evaluate_model(X_pre, y_pre, X_pre, y_pre, X_test, y_test,
                                                                       model1, criterion, optimizer1,
                                                                       is_dislplay_loss=False)
    # copy model1  to model2
    model2 = deepcopy(model1)

    # optimizer1 = optim.Adam(model1.parameters(), lr=lr)
    # optimizer2 = optim.Adam(model2.parameters(), lr=lr)
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.0)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.0)

    # Train and evaluate the model for both practices
    total_num_epochs_train = X_blocked.shape[0] // batch_size
    # total_num_epochs_train = X_blocked.shape[0]
    print("\nBlocked Practice:")
    loss_array_blocked, loss_retention_blocked, loss_test_blocked, loss_retention_array_blocked, loss_test_array_blocked = train_evaluate_model(X_blocked, y_blocked, X_blocked, y_blocked, X_test, y_test, model1, criterion, optimizer1, is_dislplay_loss=False)
    print("\nRandom Practice:")
    loss_array_random, loss_retention_random, loss_test_random, loss_retention_array_random, loss_test_array_random = train_evaluate_model(X_random, y_random, X_blocked, y_blocked, X_test, y_test, model2, criterion, optimizer2, is_dislplay_loss=False)

    # Vulnerability test
    print("\nVulnerability Test for Blocked Practice:")
    optimizer_block = optim.SGD(model1.parameters(), lr=lr, momentum=0.0)
    loss_retention_noisy_array_blocked, loss_test_noisy_array_blocked, loss_retention_pruned_array_blocked, loss_test_pruned_array_blocked, loss_retention_interf_array_blocked, loss_test_interf_array_blocked = vulnerability_test(X_blocked, y_blocked, X_test, y_test, model1, criterion, optimizer_block, num_repeat_noisy=100, num_repeat_pruned=100, num_interference_steps=100)
    print("\nVulnerability Test for Random Practice:")
    optimizer_rand = optim.SGD(model2.parameters(), lr=lr, momentum=0.0)
    loss_retention_noisy_array_random, loss_test_noisy_array_random, loss_retention_pruned_array_random, loss_test_pruned_array_random, loss_retention_interf_array_random, loss_test_interf_array_random = vulnerability_test(X_blocked, y_blocked, X_test, y_test, model2, criterion, optimizer_rand, num_repeat_noisy=100, num_repeat_pruned=100, num_interference_steps=100)


    return loss_test_pre, loss_array_blocked, loss_retention_blocked, loss_test_blocked, loss_array_random, loss_retention_random, loss_test_random, loss_retention_array_blocked, loss_test_array_blocked, loss_retention_array_random, loss_test_array_random, loss_test_array_pre, loss_retention_noisy_array_blocked, loss_test_noisy_array_blocked, loss_retention_noisy_array_random, loss_test_noisy_array_random, loss_retention_pruned_array_blocked, loss_test_pruned_array_blocked, loss_retention_pruned_array_random, loss_test_pruned_array_random, loss_retention_interf_array_blocked, loss_test_interf_array_blocked, loss_retention_interf_array_random, loss_test_interf_array_random

# set random seed for reproducibility
np.random.seed(123)
torch.manual_seed(100)

# run run_model() function in parallel
results = Parallel(n_jobs=-1)(delayed(run_model)() for _ in range(100))

# make list of each components of the results
loss_test_pre_list = [results[i][0] for i in range(len(results))]
loss_array_blocked_list = [results[i][1] for i in range(len(results))]
loss_retention_blocked_list = [results[i][2] for i in range(len(results))]
loss_test_blocked_list = [results[i][3] for i in range(len(results))]
loss_array_random_list = [results[i][4] for i in range(len(results))]
loss_retention_random_list = [results[i][5] for i in range(len(results))]
loss_test_random_list = [results[i][6] for i in range(len(results))]
loss_retention_array_blocked_list = [results[i][7] for i in range(len(results))]
loss_test_array_blocked_list = [results[i][8] for i in range(len(results))]
loss_retention_array_random_list = [results[i][9] for i in range(len(results))]
loss_test_array_random_list = [results[i][10] for i in range(len(results))]
loss_test_array_pre_list = [results[i][11] for i in range(len(results))]
loss_retention_noisy_array_blocked_list = [results[i][12] for i in range(len(results))]
loss_test_noisy_array_blocked_list = [results[i][13] for i in range(len(results))]
loss_retention_noisy_array_random_list = [results[i][14] for i in range(len(results))]
loss_test_noisy_array_random_list = [results[i][15] for i in range(len(results))]
loss_retention_pruned_array_blocked_list = [results[i][16] for i in range(len(results))]
loss_test_pruned_array_blocked_list = [results[i][17] for i in range(len(results))]
loss_retention_pruned_array_random_list = [results[i][18] for i in range(len(results))]
loss_test_pruned_array_random_list = [results[i][19] for i in range(len(results))]
loss_retention_interf_array_blocked_list = [results[i][20] for i in range(len(results))]
loss_test_interf_array_blocked_list = [results[i][21] for i in range(len(results))]
loss_retention_interf_array_random_list = [results[i][22] for i in range(len(results))]
loss_test_interf_array_random_list = [results[i][23] for i in range(len(results))]


# change list to numpy array
loss_test_pre_array = np.array(loss_test_pre_list)
loss_array_blocked_array = np.stack(loss_array_blocked_list, axis=0)
loss_retention_blocked_array = np.array(loss_retention_blocked_list)
loss_test_blocked_array = np.array(loss_test_blocked_list)
loss_array_random_array = np.stack(loss_array_random_list, axis=0)
loss_retention_random_array = np.array(loss_retention_random_list)
loss_test_random_array = np.array(loss_test_random_list)
loss_retention_array_blocked_array = np.stack(loss_retention_array_blocked_list, axis=0)
loss_test_array_blocked_array = np.stack(loss_test_array_blocked_list, axis=0)
loss_retention_array_random_array = np.stack(loss_retention_array_random_list, axis=0)
loss_test_array_random_array = np.stack(loss_test_array_random_list, axis=0)
loss_test_array_pre_array = np.stack(loss_test_array_pre_list, axis=0)
loss_retention_noisy_array_blocked = np.stack(loss_retention_noisy_array_blocked_list, axis=0)
loss_test_noisy_array_blocked = np.stack(loss_test_noisy_array_blocked_list, axis=0)
loss_retention_noisy_array_random = np.stack(loss_retention_noisy_array_random_list, axis=0)
loss_test_noisy_array_random = np.stack(loss_test_noisy_array_random_list, axis=0)
loss_retention_pruned_array_blocked = np.stack(loss_retention_pruned_array_blocked_list, axis=0)
loss_test_pruned_array_blocked = np.stack(loss_test_pruned_array_blocked_list, axis=0)
loss_retention_pruned_array_random = np.stack(loss_retention_pruned_array_random_list, axis=0)
loss_test_pruned_array_random = np.stack(loss_test_pruned_array_random_list, axis=0)
loss_retention_interf_array_blocked = np.stack(loss_retention_interf_array_blocked_list, axis=0)
loss_test_interf_array_blocked = np.stack(loss_test_interf_array_blocked_list, axis=0)
loss_retention_interf_array_random = np.stack(loss_retention_interf_array_random_list, axis=0)
loss_test_interf_array_random = np.stack(loss_test_interf_array_random_list, axis=0)


# print size of loss_retension_array_blocked_array and loss_test_array_blocked_array
print(f'Size of loss_retention_array_blocked_array: {loss_retention_array_blocked_array.shape}')
print(f'Size of loss_test_array_blocked_array: {loss_test_array_blocked_array.shape}')

# create a folder name with the learning rate with replace . with _
results_folder = f'Results_lr_{str(lr).replace(".", "_")}'
# create a folder to save the results
import os
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# save the results
np.save(results_folder + '/loss_test_pre_array.npy', loss_test_pre_array)
np.save(results_folder + '/loss_array_blocked_array.npy', loss_array_blocked_array)
np.save(results_folder + '/loss_retention_blocked_array.npy', loss_retention_blocked_array)
np.save(results_folder + '/loss_test_blocked_array.npy', loss_test_blocked_array)
np.save(results_folder + '/loss_array_random_array.npy', loss_array_random_array)
np.save(results_folder + '/loss_retention_random_array.npy', loss_retention_random_array)
np.save(results_folder + '/loss_test_random_array.npy', loss_test_random_array)
np.save(results_folder + '/loss_retention_array_blocked_array.npy', loss_retention_array_blocked_array)
np.save(results_folder + '/loss_test_array_blocked_array.npy', loss_test_array_blocked_array)
np.save(results_folder + '/loss_retention_array_random_array.npy', loss_retention_array_random_array)
np.save(results_folder + '/loss_test_array_random_array.npy', loss_test_array_random_array)
np.save(results_folder + '/loss_test_array_pre_array.npy', loss_test_array_pre_array)
np.save(results_folder + '/loss_retention_noisy_array_blocked.npy', loss_retention_noisy_array_blocked)
np.save(results_folder + '/loss_test_noisy_array_blocked.npy', loss_test_noisy_array_blocked)
np.save(results_folder + '/loss_retention_noisy_array_random.npy', loss_retention_noisy_array_random)
np.save(results_folder + '/loss_test_noisy_array_random.npy', loss_test_noisy_array_random)
np.save(results_folder + '/loss_retention_pruned_array_blocked.npy', loss_retention_pruned_array_blocked)
np.save(results_folder + '/loss_test_pruned_array_blocked.npy', loss_test_pruned_array_blocked)
np.save(results_folder + '/loss_retention_pruned_array_random.npy', loss_retention_pruned_array_random)
np.save(results_folder + '/loss_test_pruned_array_random.npy', loss_test_pruned_array_random)
np.save(results_folder + '/loss_retention_interf_array_blocked.npy', loss_retention_interf_array_blocked)
np.save(results_folder + '/loss_test_interf_array_blocked.npy', loss_test_interf_array_blocked)
np.save(results_folder + '/loss_retention_interf_array_random.npy', loss_retention_interf_array_random)
np.save(results_folder + '/loss_test_interf_array_random.npy', loss_test_interf_array_random)




