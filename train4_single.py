import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import sys

# Get list of files from directories
def get_file_list(healthy_dir, unhealthy_dir):
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".txt")]
    unhealthy_files = [os.path.join(unhealthy_dir, f) for f in os.listdir(unhealthy_dir) if f.endswith(".txt")]
    all_files = [(f, 0) for f in healthy_files] + [(f, 1) for f in unhealthy_files]  # Label: 0 for healthy, 1 for unhealthy

    print(f"Found {len(healthy_files)} healthy files and {len(unhealthy_files)} unhealthy files.")
    return all_files

# Load a batch of data from files
def load_batch(files):
    batch_data = []
    batch_labels = []

    for file_path, label in files:
        data = pd.read_csv(file_path, delimiter="\t", header=None)
        X = data.iloc[:, 1:].values  # Exclude the ID column
        y = np.full((len(data), 1), label)  # Assign the same label to all rows

        batch_data.append(X)
        batch_labels.append(y)

    # Concatenate all batch data and labels
    X_batch = np.vstack(batch_data)
    y_batch = np.vstack(batch_labels)

    print(f"Loaded a batch with {X_batch.shape[0]} samples.")
    return X_batch, y_batch

# Define the neural network model
class GeneExpressionClassifier(nn.Module):
    def __init__(self, input_size):
        super(GeneExpressionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Train the model with shuffled mixed batches
def train_model(model, optimizer, criterion, files, batch_size=1):
    model.train()
    total_loss = 0

    # Shuffle the files before each epoch
    random.shuffle(files)

    # Load data in batches
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        X_batch, y_batch = load_batch(batch_files)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.FloatTensor(y_batch)

        # Forward pass
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)

        # Print sample predictions and labels
        print(f"Sample predictions: {output[:5].detach().numpy().flatten()}")
        print(f"Sample labels: {y_tensor[:5].numpy().flatten()}")

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Processed a batch, Loss: {loss.item():.4f}")

    return total_loss / (len(files) / batch_size)

# Evaluate the model on test data
def evaluate_model(model, files, batch_size=100):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            X_batch, y_batch = load_batch(batch_files)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_batch)
            y_tensor = torch.FloatTensor(y_batch)

            # Make predictions
            predictions = (model(X_tensor) >= 0.5).float()

            # Print sample predictions and labels
            print(f"Sample predictions (binary): {predictions[:5].numpy().flatten()}")
            print(f"Sample labels: {y_tensor[:5].numpy().flatten()}")

            correct += (predictions.eq(y_tensor).sum().item())
            total += y_tensor.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Train and evaluate with a single fold (train-test split)
import sys
from sklearn.model_selection import train_test_split
from collections import defaultdict

def train_and_evaluate(files, input_size):
    # Group chunks by base file identifier
    file_groups = defaultdict(list)
    for file, label in files:
        # Extract the base identifier (everything before '_chunk')
        base_identifier = file.split('_chunk')[0]
        file_groups[base_identifier].append((file, label))
    
    # Convert the groups into a list of tuples (base_identifier, chunks)
    grouped_files = list(file_groups.items())

    # Split groups into 80% training and 20% testing
    train_groups, test_groups = train_test_split(grouped_files, test_size=0.2, random_state=int(sys.argv[1]))
    
    # Flatten the grouped data into individual files for train and test sets
    train_files = [(file, label) for _, chunks in train_groups for file, label in chunks]
    test_files = [(file, label) for _, chunks in test_groups for file, label in chunks]
    
    # Save train and test file lists
    with open('train_files.txt' + sys.argv[1], 'w') as f:
        for file, label in train_files:
            f.write(f"{file},{label}\n")
    with open('test_files.txt' + sys.argv[1], 'w') as f:
        for file, label in test_files:
            f.write(f"{file},{label}\n")

    # Define the model, loss function, and optimizer
    model = GeneExpressionClassifier(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a fixed number of epochs
    for epoch in range(1):
        print(f"\nEpoch {epoch + 1}")
        train_loss = train_model(model, optimizer, criterion, train_files)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    # Evaluate the model
    accuracy = evaluate_model(model, test_files)

    # Save the model
    torch.save(model.state_dict(), "model.pth" + sys.argv[1])
    print("Model saved to 'model.pth'")

    return accuracy

# Main execution
if __name__ == "__main__":
    healthy_dir = 'input/healthy'
    unhealthy_dir = 'input/unhealthy'

    # Get the list of files from directories
    files = get_file_list(healthy_dir, unhealthy_dir)

    # Calculate input size (exclude the ID column)

    sample_data = pd.read_csv(files[0][0], delimiter="\t", header=None)
    input_size = sample_data.shape[1] - 1  # Exclude only the ID column

    print(f"Input size detected: {input_size}")

    # Train and evaluate the model
    accuracy = train_and_evaluate(files, input_size)

# Open a file in write mode (it will create the file if it doesn't exist)
    with open("accuracy_results.txt" + sys.argv[1], "w") as f:
    # Write the accuracy to the file
         f.write(f"Test set accuracy: {accuracy * 100:.2f}%\n")

    print(f"\nTest set accuracy: {accuracy * 100:.2f}%")
