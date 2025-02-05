import numpy as np
import random
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Create an imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Ensure state has the correct shape
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)  # Ensure state has the correct shape
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Ensure next_state has the correct shape
            reward = torch.FloatTensor([reward])
            target = reward
            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            target_f = self.model(state)
            print(f"target_f shape: {target_f.shape}, action: {action}")  # Debugging print statement
            target_f = target_f.clone().detach()  # Clone and detach target_f to avoid in-place modification issues
            if target_f.size(0) > 0 and action < target_f.size(1):
                target_f[0][action] = target
            else:
                print(f"Skipping invalid assignment: target_f shape: {target_f.shape}, action: {action}")
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize DQN agent
state_size = X_train.shape[1]
action_size = 2
agent = DQN(state_size, action_size)
batch_size = 32

# Train the DQN agent
for e in range(1000):
    for i in range(len(X_train)):
        state = np.reshape(X_train[i], [1, state_size])
        action = agent.act(state)
        reward = 1 if action == y_train[i] else -1
        next_state = state
        done = True
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    if e % 10 == 0:
        print(f"Episode {e}/{1000}")

# Evaluate the agent
correct = 0
for i in range(len(X_test)):
    state = np.reshape(X_test[i], [1, state_size])
    action = agent.act(state)
    if action == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
