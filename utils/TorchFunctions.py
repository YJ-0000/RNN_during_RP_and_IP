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
    X_retention = torch.tensor(X_retention, dtype=torch.float32)
    y_retention = torch.tensor(y_retention, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Train the model
    loss_array = []
    num_batch = len(X_train) // batch_size
    num_epochs = total_num_epochs // num_batch
    epoch = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        for _ in range(num_epochs):
            epoch += 1
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss_array.append(loss.item())
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{total_num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model using retention data
    with torch.no_grad():
        outputs = model(X_retention)
        loss_retention = criterion(outputs, y_retention)
        print(f'Retention Loss: {loss_retention.item():.4f}')

    # Evaluate the model using test data
    with torch.no_grad():
        outputs = model(X_test)
        loss_test = criterion(outputs, y_test)
        print(f'Test Loss: {loss_test.item():.4f}')

    if is_dislplay_loss:
        loss_array = np.array(loss_array)
        plt.plot(loss_array)
        plt.show()

    return loss_array, loss_retention.item(), loss_test.item()

