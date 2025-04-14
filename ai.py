# ai.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class BudgetModel(nn.Module):
    def __init__(self, input_size):
        super(BudgetModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.fc(x)

def train_model():
    # Example data: You can replace this with user-uploaded data or larger dataset
    data = {
        'income': [3000, 3200, 3100, 3050, 3300],
        'rent': [1000, 1000, 1000, 1000, 1000],
        'food': [400, 420, 410, 405, 430],
        'entertainment': [150, 200, 180, 170, 210],
        'transport': [120, 130, 125, 130, 135],
        'savings': [600, 700, 650, 680, 720],
    }
    
    df = pd.DataFrame(data)

    # Split data into features and target variable
    X = df.drop(columns=['savings']).values
    y = df[['savings']].values

    # Scaling data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Convert data into PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Initialize the model, loss function, and optimizer
    model = BudgetModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Return trained model and scalers
    return model, scaler_x, scaler_y
