import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import random
import requests
import time

# Alpha Vantage API key (Replace this with your actual API key)
ALPHA_VANTAGE_API_KEY = "47A3697IO5CCP5QC"
BASE_URL = "https://www.alphavantage.co/query"

# Train a model to predict savings based on expense categories
def train_model(num_categories, monthly_income):
    # Simple neural network model for savings prediction
    model = torch.nn.Sequential(
        torch.nn.Linear(num_categories, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    # Generate fake training data for simplicity (real data should be used for training)
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

# Function to fetch and predict stock prices based on savings
def fetch_and_predict_stocks(savings_amount):
    """
    Fetches stock price predictions from Alpha Vantage API based on available savings.
    """
    # Select stock tickers for prediction
    stock_tickers = ['AAPL', 'GOOG', 'AMZN', 'TSLA', 'MSFT']
    
    # Make API calls to get the last closing prices for these stocks
    stock_predictions = {}

    for ticker in stock_tickers:
        ps = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        
        try:
            response = requests.get(BASE_URL, params=ps)
            # Log the response status and content for debugging
            print(f"Fetching data for {ticker}")
            print(f"Response Status: {response.status_code}")
            # if response.status_code == 200:
            #     print(f"Response Content: {response.json()}")
            
            data = response.json()
            # Ensure data was returned correctly and check for the right key
            if "Time Series (Daily)" in data:
                # Get the most recent trading day and its closing price
                last_trading_day = list(data["Time Series (Daily)"].keys())[0]
                last_close = data["Time Series (Daily)"][last_trading_day]["4. close"]
                stock_predictions[ticker] = float(last_close)
            else:
                # If no data returned, provide a more descriptive error message
                stock_predictions[ticker] = f"Error: No valid data for {ticker}"
        except requests.exceptions.RequestException as e:
            stock_predictions[ticker] = f"Error fetching data: {str(e)}"
        except Exception as e:
            stock_predictions[ticker] = f"Error: {str(e)}"
        
        # Sleep for a short time between requests to avoid hitting rate limits
        time.sleep(12)  # Adjust sleep time based on your API limit

    return stock_predictions

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
