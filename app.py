# app.py

import streamlit as st
import numpy as np
import torch
from ai import train_model  # Import from ai.py now

# Train the model
model, scaler_x, scaler_y = train_model()

# Set up the page configuration and theme
st.set_page_config(page_title="Centsible 💸", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: #f4f8fc;
    }
    .main {
        color: #0d1b2a;
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 40px;
        color: #1b263b;
        font-weight: bold;
    }
    .subheader {
        color: #415a77;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Displaying the title and description
st.markdown("<div class='title'>💸 Centsible</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Your smart budgeting assistant</div>", unsafe_allow_html=True)

st.write("Enter your current month's financial data:")

# Collecting user input
income = st.number_input("Monthly Income ($)", min_value=0, value=3000)
rent = st.number_input("Rent/Mortgage ($)", min_value=0, value=1000)
food = st.number_input("Food ($)", min_value=0, value=400)
entertainment = st.number_input("Entertainment ($)", min_value=0, value=150)
transport = st.number_input("Transport ($)", min_value=0, value=120)

# When the user clicks the button to calculate the budget
if st.button("Suggest Budget"):
    user_input = np.array([[income, rent, food, entertainment, transport]])
    user_scaled = scaler_x.transform(user_input)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
    
    # Make a prediction for savings
    predicted_value = model(user_tensor).item()
    predicted_savings = scaler_y.inverse_transform([[predicted_value]])

    # Calculate suggested spending based on predicted savings
    suggested_savings = float(predicted_savings[0][0])
    expenses = rent + food + entertainment + transport
    suggested_spending = income - suggested_savings

    # Show the result to the user
    st.success(f"✅ Based on your data, you could aim to save **${suggested_savings:.2f}** this month.")
    st.write(f"🔧 That leaves **${suggested_spending:.2f}** for your total spending.")
    
    st.markdown("---")
    st.subheader("💡 Suggested Budget Breakdown")
    st.markdown(f"""
    - 🏠 **Rent**: ${rent}
    - 🍽️ **Food**: ${food}
    - 🎉 **Entertainment**: ${entertainment}
    - 🚗 **Transport**: ${transport}
    - 💾 **Savings**: **${suggested_savings:.2f}**
    """)
