import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import random

# Train a model to predict savings based on expense categories
def train_model(num_categories, monthly_income):
    # Dummy model - replace with an actual model for better predictions
    model = torch.nn.Sequential(
        torch.nn.Linear(num_categories, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    # Generate some fake training data for simplicity
    # In real use, you would train this model on real historical data
    X_train = np.random.rand(100, num_categories)
    y_train = np.random.rand(100, 1) * monthly_income  # Predicted savings between 0 and monthly_income

    # Normalize the data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    # Fit the scalers and transform the training data
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Train the model with fake data
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    
    for epoch in range(100):  # Dummy epochs for training
        model.train()
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    return model, scaler_x, scaler_y

# Function to predict stock prices based on savings
def fetch_and_predict_stocks(savings_amount):
    # Dummy stock predictions - replace with actual stock prediction logic
    stock_tickers = ['AAPL', 'GOOG', 'AMZN', 'TSLA', 'MSFT']
    predicted_prices = {ticker: random.uniform(100, 5000) for ticker in stock_tickers}
    
    return predicted_prices

# Adjust non-necessity expenses to maximize savings
def adjust_non_necessities_for_savings(non_necessities, suggested_spending):
    """
    This function takes in the non-necessities expenses and adjusts them based on
    the available budget for non-necessities, ensuring more money is allocated for savings.
    """
    non_necessity_total = sum([value for _, value in non_necessities])
    
    # If total non-necessity spending exceeds the suggested spending, adjust it
    if non_necessity_total > suggested_spending:
        excess = non_necessity_total - suggested_spending
        # Adjust the non-necessities evenly by cutting down the excess amount
        adjusted_non_necessities = []
        for category, value in non_necessities:
            adjustment = value - (excess * (value / non_necessity_total))
            adjusted_non_necessities.append((category, max(adjustment, 0)))  # Ensure no negative values
        return adjusted_non_necessities
    else:
        return non_necessities  # No adjustment needed if within budget

# Suggest a budget for the user based on their monthly income and expenses
def suggest_budget(necessities, non_necessities, monthly_income):
    """
    This function calculates a suggested budget based on the monthly income,
    necessary expenses, and non-necessities, adjusting for a reasonable savings amount.
    """
    # Calculate the total necessary expenses
    total_necessities = sum([value for _, value in necessities])
    
    # Predict the savings based on model prediction and scale to monthly income
    predicted_savings = min(monthly_income - total_necessities, monthly_income * 0.3)  # Cap at 30% of income
    
    # Calculate the suggested spending for non-necessities
    suggested_spending = monthly_income - total_necessities - predicted_savings
    
    # Adjust the non-necessities based on available spending
    adjusted_non_necessities = adjust_non_necessities_for_savings(non_necessities, suggested_spending)

    return {
        "suggested_savings": predicted_savings,
        "suggested_spending": suggested_spending,
        "adjusted_non_necessities": adjusted_non_necessities
    }
