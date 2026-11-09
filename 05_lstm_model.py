import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

# Load preprocessed data
data = np.load("artifacts/preprocessed_data.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.long)
X_valid = torch.tensor(data["X_valid"], dtype=torch.long)

y_train = torch.tensor(data["y_train"], dtype=torch.long)
y_valid = torch.tensor(data["y_valid"], dtype=torch.long)

# Load vocabulary
with open("artifacts/vocabulary.json", "r") as file:
    vocab = json.load(file)

print("Training data:", X_train.shape)
print("Validation data:", X_valid.shape)
print("Vocabulary size:", len(vocab))

# LSTM Model

class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):

        super(LSTMModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    # Forward propagation
    def forward(self, x):

        embedded = self.embedding(x)

        output, (hidden, cell) = self.lstm(embedded)

        # Find the last non-padding token
        lengths = (x != 0).sum(dim=1)

        last_output = output[
            torch.arange(x.size(0), device=x.device),
            lengths - 1
        ]

        prediction = self.fc(last_output)

        return prediction


# Model parameters
vocab_size = len(vocab)
embedding_dim = 64
hidden_dim = 128
num_classes = 3

# Create LSTM model
model = LSTMModel(
    vocab_size,
    embedding_dim,
    hidden_dim,
    num_classes
)

print("\nLSTM Model:")
print(model)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training settings
EPOCHS = 10
BATCH_SIZE = 64

print("\nLSTM Training Components:")
print("Loss Function:", criterion)
print("Optimizer:", optimizer)
print("Epochs:", EPOCHS)
print("Batch Size:", BATCH_SIZE)

# Create training dataset
# Create training dataset
train_dataset = TensorDataset(X_train, y_train)

# Create validation dataset
valid_dataset = TensorDataset(X_valid, y_valid)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoaders created successfully!")
print("Training batches:", len(train_loader))
print("Validation batches:", len(valid_loader))

# Set device
device = torch.device("cpu")

model = model.to(device)

# Training loop
for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Make predictions
        outputs = model(batch_X)

        # Calculate loss
        loss = criterion(outputs, batch_y)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        total_train_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        correct_train += (predictions == batch_y).sum().item()
        total_train += batch_y.size(0)

    # Training metrics
    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation
    model.eval()

    total_valid_loss = 0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():

        for batch_X, batch_y in valid_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            total_valid_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct_valid += (predictions == batch_y).sum().item()
            total_valid += batch_y.size(0)

    valid_loss = total_valid_loss / len(valid_loader)
    valid_accuracy = correct_valid / total_valid

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Training Loss: {train_loss:.4f} "
        f"Training Accuracy: {train_accuracy:.4f} "
        f"Validation Loss: {valid_loss:.4f} "
        f"Validation Accuracy: {valid_accuracy:.4f}"
    )
    from sklearn.metrics import classification_report, confusion_matrix

# Evaluate LSTM model
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in valid_loader:

        outputs = model(batch_X)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

# Classification report
print("\nLSTM Classification Report:")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Bearish", "Bullish", "Neutral"],
    digits=4
))

# Confusion matrix
print("\nLSTM Confusion Matrix:")

print(confusion_matrix(all_labels, all_predictions))

import os

os.makedirs("models", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/lstm_model.pth"
)

print("LSTM model saved successfully!")


