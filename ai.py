import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

def train_model(input_size, monthly_income, num_samples=100):
    """
    Generate synthetic data where each feature is an expense category,
    and the target is the remaining savings (monthly_income - sum(expenses)).
    """
    np.random.seed(42)

    max_expense = monthly_income * 0.6 / input_size
    X = np.random.uniform(low=0.0, high=max_expense, size=(num_samples, input_size))
    y = monthly_income - np.sum(X, axis=1, keepdims=True)

    # Scale features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

    # Build model
    model = BudgetModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler_x, scaler_y

def fetch_and_predict_stocks(savings):
    """
    Mock function to return stock predictions based on savings.
    """
    stock_suggestions = {
        'AAPL': torch.tensor([145.00]),
        'GOOG': torch.tensor([2725.00]),
        'AMZN': torch.tensor([3345.00]),
    }
    return stock_suggestions
