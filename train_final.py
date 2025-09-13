import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random

# ------------------------------
# Data utilities
# ------------------------------
def get_file_list(healthy_dir, unhealthy_dir):
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith(".txt")]
    unhealthy_files = [os.path.join(unhealthy_dir, f) for f in os.listdir(unhealthy_dir) if f.endswith(".txt")]
    all_files = [(f, 0) for f in healthy_files] + [(f, 1) for f in unhealthy_files]  # 0=healthy, 1=unhealthy
    print(f"Found {len(healthy_files)} healthy files and {len(unhealthy_files)} unhealthy files.")
    return all_files

def load_batch(files):
    batch_data = []
    batch_labels = []
    batch_file_names = []
    for file_path, label in files:
        data = pd.read_csv(file_path, delimiter="\t", header=None)
        X = data.iloc[:, 1:].values  # exclude ID column
        y = np.full((len(data), 1), label)
        batch_data.append(X)
        batch_labels.append(y)
        file_name = os.path.basename(file_path)
        batch_file_names.extend([file_name] * len(data))
    X_batch = np.vstack(batch_data)
    y_batch = np.vstack(batch_labels)
    print(f"Loaded a batch with {X_batch.shape[0]} samples.")
    return X_batch, y_batch, batch_file_names

# ------------------------------
# Model
# ------------------------------
class GeneExpressionClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
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

# ------------------------------
# Training loop (uses all files)
# ------------------------------
def train_single_model(model, optimizer, criterion, files, batch_size=1):
    model.train()
    total_loss = 0.0
    # shuffle file order each epoch
    random.shuffle(files)

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        X_batch, y_batch, _ = load_batch(batch_files)
        X_tensor = torch.FloatTensor(X_batch)
        y_tensor = torch.FloatTensor(y_batch)

        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {i // batch_size + 1}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(len(files) / batch_size, 1)
    return avg_loss

def train_final_models(files, input_size, num_models=5, epochs=1, lr=1e-5, batch_size=1, out_prefix="final_model"):
    """
    Train N models on ALL data (no validation/testing). Saves weights to {out_prefix}_{i}.pth
    """
    models = []
    for i in range(num_models):
        print(f"\nTraining final model {i + 1}/{num_models} with seed {i}")
        # set seeds for reproducibility
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        model = GeneExpressionClassifier(input_size=input_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} / {epochs}  (model {i + 1})")
            train_loss = train_single_model(model, optimizer, criterion, files, batch_size=batch_size)
            print(f"Model {i + 1}, Epoch {epoch + 1}, Avg Loss: {train_loss:.4f}")

        save_path = f"{out_prefix}_{i+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model {i + 1} to {save_path}")
        models.append(model)

    print("\nDone. Trained and saved all final models.")
    return models

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    healthy_dir = 'input/healthy'
    unhealthy_dir = 'input/unhealthy'

    files = get_file_list(healthy_dir, unhealthy_dir)

    # detect input size from the first file
    sample_data = pd.read_csv(files[0][0], delimiter="\t", header=None)
    input_size = sample_data.shape[1] - 1
    print(f"Input size detected: {input_size}")

    # Train five final models on ALL data (no CV)
    _ = train_final_models(
        files=files,
        input_size=input_size,
        num_models=5,      # five final models
        epochs=1,          # change if you want more epochs
        lr=1e-5,
        batch_size=1,
        out_prefix="final_model"
    )

    # Note: no evaluation here by design (you asked to use all data for training).
    # If you later want predictions on some dataset, load these .pth files and run inference.

