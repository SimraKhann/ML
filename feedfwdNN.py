import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(X_train.columns), 128),  # Adjust input dimension according to your data
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Adjust output dimension for your classification task
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities




def create_dataloader(X, y, batch_size):
    # Convert X and y to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    # Create a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # iterates over the dataloader object
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error. passes each batch to the model; results in [batch_size] worth of predictions
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    avg_loss = test_loss / num_batches
    accuracy = correct /size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return(accuracy*100) # returning accuracy percentage
# Specify the path to your TSV file
data = pd.read_csv("/home/ibab/SEM_4/project/data/gvcf/machineLearning/server_final_updated_data_with_target_all.tsv", delimiter="\t", usecols=lambda col: col != 0)
data.drop("SampleName", axis=1, inplace=True)
data.fillna(0, inplace=True)
data.reset_index(drop=True, inplace=True)
# Separate features and target variable
X = data.drop("target", axis=1)
y = data["target"] 


# Define parameters
batch_size = 64
k_folds = 10
num_epochs = 200
torch.manual_seed(42)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

    # Split data into training and test sets (e.g., 70%/30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # K-fold for training data
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

final_scores = []

for fold, (train_index, val_index) in enumerate(kfold.split(X_train)):	
    print(f'FOLD {fold}')
    print('--------------------------------')

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Apply He initialization to all layers
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(2), nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        # Create Dataloaders
    train_dataloader = create_dataloader(X_train_fold, y_train_fold, batch_size)
    val_dataloader = create_dataloader(X_val_fold, y_val_fold, batch_size)

        # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
            # Print epoch
        print(f'Starting epoch {epoch + 1}')

            # Train the model
        train(train_dataloader, model, loss_fn, optimizer)

        # Evaluate on validation set after each fold
    accuracy = test(val_dataloader, model, loss_fn)

    final_scores.append(float(accuracy))  # Extract accuracy value

print("Completed 10-Fold CV\n")
print(f"Average accuracy across folds: {np.mean(final_scores):.2f}%\n\n")

    #Final Evaluation
print("Final Evaluation on test dataset")
test_dataloader = create_dataloader(X_test,y_test,batch_size)

    # Final evaluation on test set
model.eval()
correct = 0
size = len(test_dataloader.dataset)
with torch.no_grad():
    for matx, label in test_dataloader:
        label = label.to(device)
        matx = matx.to(device)
        pred = model(matx)

        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

accuracy = correct / size
print(f"Test Accuracy: {accuracy * 100:.2f}%")
