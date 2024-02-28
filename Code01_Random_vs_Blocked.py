import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from utils.NetworkModels import MotorLearningRNN
from utils.DataGen import generate_synthetic_data
from utils.TorchFunctions import train_evaluate_model
from joblib import Parallel, delayed

def run_model():
    # Hyperparameters
    input_size = 7  # Number of features in the input. For motor learning, this could represent different aspects of the task or context.
    hidden_size = 7  # Number of features in the hidden state
    output_size = 4  # For simplicity, let's say we're predicting 4 different outcomes related to motor learning


    # Generate synthetic data for both practices
    X_blocked, y_blocked, X_random, y_random = generate_synthetic_data(6, input_noise=True)
    print(X_blocked.shape)
    print(y_blocked.shape)

    X_pre, y_pre, X_pre_random, y_pre_random = generate_synthetic_data(10, input_noise=True)
    _, _, X_test, y_test = generate_synthetic_data(200, input_noise=True)

    # Instantiate the model
    model1 = MotorLearningRNN(input_size, hidden_size, output_size, num_outputs=input_size)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.005)
    train_evaluate_model(X_pre, y_pre, X_test, y_test, model1, criterion, optimizer1, total_num_epochs=30000, is_dislplay_loss=False)
    _, _, loss_test_pre = train_evaluate_model(X_pre_random, y_pre_random, X_test, y_test, model1, criterion, optimizer1, total_num_epochs=30000, is_dislplay_loss=False)
    # copy model1  to model2
    model2 = deepcopy(model1)

    optimizer1 = optim.Adam(model1.parameters(), lr=0.005)
    optimizer2 = optim.Adam(model1.parameters(), lr=0.005)

    # Train and evaluate the model for both practices
    print("\nBlocked Practice:")
    loss_array_blocked, loss_retention_blocked, loss_test_blocked = train_evaluate_model(X_blocked, y_blocked, X_test, y_test, model1, criterion, optimizer1, total_num_epochs=30000, is_dislplay_loss=False)
    print("\nRandom Practice:")
    loss_array_random, loss_retention_random, loss_test_random = train_evaluate_model(X_random, y_random, X_test, y_test, model2, criterion, optimizer2, total_num_epochs=30000, is_dislplay_loss=False)

    return loss_test_pre, loss_array_blocked, loss_retention_blocked, loss_test_blocked, loss_array_random, loss_retention_random, loss_test_random

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

# change list to numpy array
loss_test_pre_array = np.array(loss_test_pre_list)
loss_array_blocked_array = np.stack(loss_array_blocked_list, axis=0)
loss_retention_blocked_array = np.array(loss_retention_blocked_list)
loss_test_blocked_array = np.array(loss_test_blocked_list)
loss_array_random_array = np.stack(loss_array_random_list, axis=0)
loss_retention_random_array = np.array(loss_retention_random_list)
loss_test_random_array = np.array(loss_test_random_list)

results_folder = 'results'
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


# paired t-test for loss_retention_blocked_array and loss_retention_random_array
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(loss_retention_blocked_array, loss_retention_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')

# paired t-test for loss_test_blocked_array and loss_test_random_array
t_stat, p_value = ttest_rel(loss_test_blocked_array, loss_test_random_array)
print(f't-statistic: {t_stat}, p-value: {p_value}')


# Plot the results of the mean of the loss
plt.figure()
plt.plot(loss_array_blocked_array.mean(axis=0), label='Blocked')
plt.plot(loss_array_random_array.mean(axis=0), label='Random')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Test Loss')
plt.show()

# Grouped bar chart the results of the mean and error bar of the loss for pre, retention, and test
plt.figure()
barWidth = 0.25
r1 = np.arange(3)
r2 = [x + barWidth for x in r1]
plt.bar(r1, [0, loss_retention_blocked_array.mean(), loss_test_blocked_array.mean()], yerr=[0, loss_retention_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_blocked_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Blocked')
plt.bar(r2, [loss_test_pre_array.mean(), loss_retention_random_array.mean(), loss_test_random_array.mean()], yerr=[loss_test_pre_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_retention_random_array.std()/np.sqrt(np.size(loss_test_pre_array)), loss_test_random_array.std()/np.sqrt(np.size(loss_test_pre_array))], width=barWidth, label='Random')
plt.xticks([r + barWidth for r in range(3)], ['Pre', 'Retention', 'Test'])
plt.ylabel('Loss')
plt.legend()
plt.title('Retention Loss')
plt.show()
