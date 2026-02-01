import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
from utils.DataGen import generate_synthetic_data


# Function to train and evaluate the model
def train_evaluate_model(X, y, X_retention, y_retention, X_test, y_test ,model, criterion, optimizer, batch_size=20, is_dislplay_loss=True):

    # check X, y are empty
    if X.shape[0] != 0 or y.shape[0] != 0:
        # Split the data into training and testing sets
        X_train = torch.tensor(X, dtype=torch.float32)
        y_train = torch.tensor(y, dtype=torch.float32)
        X_retention = torch.tensor(X_retention, dtype=torch.float32)
        y_retention = torch.tensor(y_retention, dtype=torch.float32)

        # Train the model
        loss_array = []
        num_batch = len(X_train) // batch_size

        # print length of X_train, batch size, number of batches, and number of epochs per batch
        print(f'Length of X_train: {len(X_train)}, Batch size: {batch_size}, Number of batches: {num_batch}')

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            # Forward pass
            outputs = model(X_batch)

            # reshape y_batch for treat each sequential output as independent
            y_batch_view = y_batch.view(-1, y_batch.shape[2])
            outputs_view = outputs.view(-1, outputs.shape[2])

            y_batch_class = torch.argmax(y_batch_view, dim=1)

            loss = criterion(outputs_view, y_batch_class)
            loss_array.append(loss.item())
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (epoch + 1) % 10 == 0:
            #     print(f'Epoch [{epoch + 1}/{total_num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model using retention data
        with torch.no_grad():
            outputs = model(X_retention)
            # reshape y_retention and outputs for treat each sequential output as independent
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_retention = criterion(outputs_view, y_retention_class)
            print(f'Retention Loss: {loss_retention.item():.4f}')
            # calculate loss for each output separately
            print('y_retention shape:', y_retention.shape)
            loss_retention_array = []
            for i in range(y_retention.shape[0]):
                loss_retention_i = criterion(outputs[i, :, :], y_retention[i, :, :])
                loss_retention_array.append(loss_retention_i.item())
            # check the average of loss_retention_array is almost same as loss_retention
            print(f'Retention Loss (each output): {loss_retention:.4f}, Average: {np.mean(loss_retention_array):.4f}')
    else:
        loss_array = []
        loss_retention = torch.tensor(0.0)
        loss_retention_array = []
        print('No training data provided.')

    # Evaluate the model using transfer data
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test)
        # reshape y_test and outputs for treat each sequential output as independent
        y_test_view = y_test.view(-1, y_test.shape[2])
        y_test_class = torch.argmax(y_test_view, dim=1)
        outputs_view = outputs.view(-1, outputs.shape[2])
        # calculate loss
        loss_test = criterion(outputs_view, y_test_class)
        print(f'Test Loss: {loss_test.item():.4f}')
        # calculate loss for each output separately
        loss_test_array = []
        for i in range(y_test.shape[0]):
            loss_test_i = criterion(outputs[i, :, :], y_test[i, :, :])
            loss_test_array.append(loss_test_i.item())
        # check the average of loss_test_array is almost same as loss_test
        print(f'Test Loss (each output): {loss_test:.4f}, Average: {np.mean(loss_test_array):.4f}')

    if is_dislplay_loss:
        loss_array = np.array(loss_array)
        plt.plot(loss_array)
        plt.show()

    return loss_array, loss_retention.item(), loss_test.item(), loss_retention_array, loss_test_array


def vulnerability_test(X_retention, y_retention, X_test, y_test, model, criterion, optimizer,
                       num_repeat_noisy=10, num_repeat_pruned=10, num_interference_steps=100,
                       batch_size=20):
    # Prepare retention and test data
    X_retention = torch.tensor(X_retention[- (X_retention.shape[0] // 3):], dtype=torch.float32)
    y_retention = torch.tensor(y_retention[- (y_retention.shape[0] // 3):], dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # print shapes of retention and test data
    print(f'X_retention shape: {X_retention.shape}, y_retention shape: {y_retention.shape}')

    ### (1) test the model vulnerability by adding noise to the weights
    # copy the model
    model_noisy = deepcopy(model)
    # noise parameter
    noise_std = 0.1
    ## repeat add noise and test a given number of times
    loss_retention_noisy_array = []
    loss_test_noisy_array = []
    for repeat in range(num_repeat_noisy):
        # add noise to the weights
        if repeat > 0:
            # initial repeat is for retaining performance before perturbation
            with torch.no_grad():
                for param in model_noisy.parameters():
                    # generate gaussian noise
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)
        # Evaluate the model using retention data
        with torch.no_grad():
            outputs = model_noisy(X_retention)
            # reshape y_retention and outputs for treat each sequential output as independent
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_retention_noisy = criterion(outputs_view, y_retention_class)
            loss_retention_noisy_array.append(loss_retention_noisy.item())
        # Evaluate the model using transfer data
        with torch.no_grad():
            outputs = model_noisy(X_test)
            # reshape y_test and outputs for treat each sequential output as independent
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_test_noisy = criterion(outputs_view, y_test_class)
            loss_test_noisy_array.append(loss_test_noisy.item())

    # print all the noisy retention losses and test losses
    print(f'Noisy Retention Losses: {loss_retention_noisy_array}')
    print(f'Noisy Test Losses: {loss_test_noisy_array}')

    ### (2) test the model vulnerability by adversarial Weight Rounding (pruning)
    # copy the model
    model_pruned = deepcopy(model)
    # pruning parameter
    pruning_percent = 0.05
    ## repeat pruning and test a given number of times
    loss_retention_pruned_array = []
    loss_test_pruned_array = []
    for repeat in range(num_repeat_pruned):
        # prune the weights
        if repeat > 0:
            # initial repeat is for retaining performance before perturbation
            with torch.no_grad():
                for param in model_pruned.parameters():
                    # randomly set a percentage of weights to zero
                    mask = torch.rand_like(param) > pruning_percent
                    param.mul_(mask)
        # Evaluate the model using retention data
        with torch.no_grad():
            outputs = model_pruned(X_retention)
            # reshape y_retention and outputs for treat each sequential output as independent
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_retention_pruned = criterion(outputs_view, y_retention_class)
            loss_retention_pruned_array.append(loss_retention_pruned.item())
        # Evaluate the model using transfer data
        with torch.no_grad():
            outputs = model_pruned(X_test)
            # reshape y_test and outputs for treat each sequential output as independent
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_test_pruned = criterion(outputs_view, y_test_class)
            loss_test_pruned_array.append(loss_test_pruned.item())

    # print all the pruned retention losses and test losses
    print(f'Pruned Retention Losses: {loss_retention_pruned_array}')
    print(f'Pruned Test Losses: {loss_test_pruned_array}')

    # (3) Interferene test for learning new sequences
    _, _, X_interference, y_interference = generate_synthetic_data(num_sequence_types=1, num_samples=batch_size*num_interference_steps, input_noise=True)
    loss_retention_interference_array = []
    loss_test_interference_array = []
    for i in range(-batch_size, len(X_interference), batch_size):
        if i >= 0:
            # initial repeat is for retaining performance before perturbation
            X_batch = torch.tensor(X_interference[i:i + batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_interference[i:i + batch_size], dtype=torch.float32)
            # Forward pass
            outputs = model(X_batch)

            # reshape y_batch for treat each sequential output as independent
            y_batch_view = y_batch.view(-1, y_batch.shape[2])
            outputs_view = outputs.view(-1, outputs.shape[2])

            y_batch_class = torch.argmax(y_batch_view, dim=1)

            loss = criterion(outputs_view, y_batch_class)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate the model using retention data
        with torch.no_grad():
            outputs = model(X_retention)
            # reshape y_retention and outputs for treat each sequential output as independent
            y_retention_view = y_retention.view(-1, y_retention.shape[2])
            y_retention_class = torch.argmax(y_retention_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_retention_interference = criterion(outputs_view, y_retention_class)
            loss_retention_interference_array.append(loss_retention_interference.item())
        # Evaluate the model using transfer data
        with torch.no_grad():
            outputs = model(X_test)
            # reshape y_test and outputs for treat each sequential output as independent
            y_test_view = y_test.view(-1, y_test.shape[2])
            y_test_class = torch.argmax(y_test_view, dim=1)
            outputs_view = outputs.view(-1, outputs.shape[2])
            # calculate loss
            loss_test_interference = criterion(outputs_view, y_test_class)
            loss_test_interference_array.append(loss_test_interference.item())

    print(f'Interference Retention Losses: {loss_retention_interference_array}')
    print(f'Interference Test Losses: {loss_test_interference_array}')

    return loss_retention_noisy_array, loss_test_noisy_array, loss_retention_pruned_array, loss_test_pruned_array, loss_retention_interference_array, loss_test_interference_array


