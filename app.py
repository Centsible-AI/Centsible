import streamlit as st
import numpy as np
import torch
from ai import train_model, fetch_and_predict_stocks

# Set up the page configuration
st.set_page_config(page_title="Centsible 💸", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        color: #1E90FF;
        font-weight: bold;
        text-align: center;
    }
    body {
        background-color: white;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title section
st.markdown("<div class='title'>Centsible</div>", unsafe_allow_html=True)

# Input for yearly income
st.write("Enter your yearly income:")
yearly_income = st.number_input("Yearly Income ($)", min_value=0.0, value=0.0, step=0.01)
monthly_income = yearly_income / 12
st.write(f"Your monthly income is: **${monthly_income:.2f}**")

# Dynamic number of expense category inputs
st.write("Enter your custom expense categories:")
category_count = st.number_input("Number of Expense Categories", min_value=1, value=1, step=1)

categories = []
# Create one input row per category
for i in range(int(category_count)):
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input(f"Category {i+1} Name", key=f"name_{i}")
    with col2:
        value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"value_{i}")
    # Only add the category if a name is provided
    if name:
        categories.append((name, value))

# Display entered categories
if categories:
    st.write("Your entered expense categories:")
    for category, value in categories:
        st.write(f"- {category}: ${value:.2f}")

if st.button("Suggest Budget"):
    # Ensure all categories (as specified by number input) are filled
    if not categories or len(categories) != int(category_count):
        st.error("Please add all categories before suggesting a budget.")
    else:
        # Build user input vector from the expense values
        values = [val for _, val in categories]
        user_input = np.array([values])  # Shape: (1, number_of_categories)
        
        # Train the model dynamically with the number of categories and the provided monthly income.
        model, scaler_x, scaler_y = train_model(len(categories), monthly_income)
        
        # Scale the user input and predict using the trained model
        user_scaled = scaler_x.transform(user_input)
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
        predicted_value = model(user_tensor).item()
        
        # Convert the prediction back to the original savings scale
        predicted_savings = scaler_y.inverse_transform([[predicted_value]])
        suggested_savings = float(predicted_savings[0][0])
        suggested_spending = monthly_income - suggested_savings
        
        st.success(f"✅ You could aim to save **${suggested_savings:.2f}** this month.")
        st.write(f"🔧 That leaves **${suggested_spending:.2f}** for spending.")
        
        st.markdown("---")
        st.subheader("💡 Suggested Budget Breakdown")
        for category, value in categories:
            st.write(f"- **{category}**: ${value:.2f}")
        st.write(f"- 💾 **Savings**: **${suggested_savings:.2f}**")
        
        # Display mock stock suggestions based on the predicted savings
        stock_suggestions = fetch_and_predict_stocks(suggested_savings)
        st.markdown("---")
        st.subheader("📈 Stock Suggestions Based on Your Budget")
        for ticker, predicted_price in stock_suggestions.items():
            st.write(f"**{ticker}**: Predicted Price for Next Day: ${predicted_price.item():.2f}")
