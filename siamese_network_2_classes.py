import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np

# Define the base network
class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

# Define the Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork(input_dim)

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2

# Custom dataset for Siamese network
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        return torch.tensor(pair[0], dtype=torch.float32), torch.tensor(pair[1], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Function to create pairs of data points
def create_pairs(X, y):
    pairs = []
    labels = []
    n = len(y)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append([X[i], X[j]])
            labels.append(1 if y[i] == y[j] else 0)
    return np.array(pairs), np.array(labels)

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Load your dataset
# Assuming X contains features and y contains labels
# X, y = load_your_data()

# For demonstration purposes, let's create dummy data
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(0, 2, 1000)

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalanced dataset using Random Undersampling
class_0 = X_scaled[y == 0]
class_1 = X_scaled[y == 1]

if len(class_0) > len(class_1):
    class_0 = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)
else:
    class_1 = resample(class_1, replace=False, n_samples=len(class_0), random_state=42)

X_resampled = np.vstack((class_0, class_1))
y_resampled = np.hstack((np.zeros(len(class_0)), np.ones(len(class_1))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create pairs of data points
pairs_train, labels_train = create_pairs(X_train, y_train)
pairs_test, labels_test = create_pairs(X_test, y_test)

# Create datasets and dataloaders
train_dataset = SiameseDataset(pairs_train, labels_train)
test_dataset = SiameseDataset(pairs_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = SiameseNetwork(input_dim)
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input1, input2, label = batch
        optimizer.zero_grad()
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in test_loader:
        input1, input2, label = batch
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, label)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader):.4f}')
