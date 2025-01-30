import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic imbalanced dataset (e.g., transactional data)
X, y = make_classification(
    n_samples=10000,  # Total samples
    n_features=5,     # Number of numerical features
    n_informative=5,  # Informative features
    n_redundant=0,    # No redundant features
    n_classes=2,      # Binary classification
    weights=[0.99, 0.01],  # Imbalanced classes (99% majority, 1% minority)
    random_state=42
)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (important for neural networks)
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Hyperparameters
input_size = X_train.shape[1]  # Number of features
hidden_size = 32               # Hidden layer size
output_size = 2                # Binary classification
learning_rate = 0.01
num_epochs = 100
gamma = 0.99  # Discount factor

# Initialize policy network and optimizer
policy_net = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Reward function
def get_reward(prediction, true_label):
    if prediction == true_label:
        if true_label == 1:  # Minority class (e.g., fraud)
            return 10.0
        else:  # Majority class (e.g., legitimate)
            return 1.0
    else:
        if true_label == 1:  # Misclassified minority class
            return -5.0
        else:  # Misclassified majority class
            return -1.0

# Training loop
for epoch in range(num_epochs):
    policy_net.train()
    optimizer.zero_grad()

    # Sample a batch of data
    indices = np.random.choice(len(X_train), size=256, replace=True)  # Mini-batch
    states = X_train[indices]
    true_labels = y_train[indices]

    # Forward pass: get action probabilities
    action_probs = policy_net(states)
    actions = torch.multinomial(action_probs, num_samples=1).squeeze()

    # Compute rewards
    rewards = torch.tensor([get_reward(actions[i].item(), true_labels[i].item()) for i in range(len(actions))])

    # Compute policy loss
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    loss = -torch.sum(log_probs.squeeze() * rewards)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    if (epoch + 1) % 10 == 0:
        policy_net.eval()
        with torch.no_grad():
            test_probs = policy_net(X_test)
            test_preds = torch.argmax(test_probs, dim=1)
            f1 = f1_score(y_test.numpy(), test_preds.numpy(), average="binary")
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, F1-Score: {f1:.4f}")

# Final evaluation
policy_net.eval()
with torch.no_grad():
    test_probs = policy_net(X_test)
    test_preds = torch.argmax(test_probs, dim=1)
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), test_preds.numpy(), target_names=["Legitimate", "Fraud"]))
