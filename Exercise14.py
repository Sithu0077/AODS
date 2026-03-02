# ======================================
# Simple MLP for Binary Classification
# ======================================

import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset
X = torch.rand(100, 10)
y = torch.randint(0, 2, (100,))

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training
epochs = 10

for epoch in range(epochs):
    
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / len(y)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {loss.item():.4f} "
          f"Accuracy: {accuracy*100:.2f}%")