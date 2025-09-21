import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Function to train and evaluate the model
def train_evaluate_model(X, y, X_test, y_test ,model, criterion, optimizer, batch_size=20, total_num_epochs=10000, is_dislplay_loss=True):
    # Split the data into training and testing sets
    _, X_retention, _, y_retention = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)
    # X_retention = torch.tensor(X_retention, dtype=torch.float32)
    # y_retention = torch.tensor(y_retention, dtype=torch.float32)
    X_retention = X_train
    y_retention = y_train
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

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

    # Evaluate the model using transfer data
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

