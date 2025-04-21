import streamlit as st
import numpy as np
import torch
from ai import train_model, fetch_and_predict_stocks

# --- Page setup ---
st.set_page_config(page_title="Centsible 💸", layout="centered")

# --- Init session state ---
if "users" not in st.session_state:
    st.session_state.users = {
        "alice": "password123",
        "bob": "securepass",
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# --- Auth logic ---
def login(username, password):
    users = st.session_state.users
    if username in users and users[username] == password:
        st.session_state.logged_in = True
        st.session_state.username = username
    else:
        st.error("Invalid username or password.")

def signup(username, password):
    users = st.session_state.users
    if username in users:
        st.error("Username already exists.")
    elif not username or not password:
        st.warning("Username and password cannot be empty.")
    else:
        users[username] = password
        st.success("Account created! You can now log in.")
        st.balloons()

# --- Login / Signup UI ---
if not st.session_state.logged_in:
    st.title("🔐 Welcome to Centsible")

    mode = st.radio("Choose an option:", ["Login", "Sign Up"], horizontal=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Login":
        if st.button("Login"):
            login(username, password)
    else:
        if st.button("Sign Up"):
            signup(username, password)

    st.stop()

# --- Main App After Login ---
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        color: #1E90FF;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>Centsible</div>", unsafe_allow_html=True)
st.write(f"👋 Welcome, **{st.session_state.username}**")

# Logout button
if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# Income input
st.write("Enter your yearly income:")
yearly_income = st.number_input("Yearly Income ($)", min_value=0.0, value=0.0, step=0.01)
monthly_income = yearly_income / 12
st.write(f"Your monthly income is: **${monthly_income:.2f}**")

# Expense categories
st.write("Enter your custom expense categories:")
category_count = st.number_input("Number of Expense Categories", min_value=1, value=1, step=1)

categories = []
for i in range(int(category_count)):
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input(f"Category {i+1} Name", key=f"name_{i}")
    with col2:
        value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"value_{i}")
    if name:
        categories.append((name, value))

# Show entered categories
if categories:
    st.write("Your entered expense categories:")
    for category, value in categories:
        st.write(f"- {category}: ${value:.2f}")

# Suggest budget
if st.button("Suggest Budget"):
    if not categories or len(categories) != int(category_count):
        st.error("Please add all categories before suggesting a budget.")
    else:
        values = [val for _, val in categories]
        user_input = np.array([values])

        model, scaler_x, scaler_y = train_model(len(categories), monthly_income)
        user_scaled = scaler_x.transform(user_input)
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
        predicted_value = model(user_tensor).item()

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

        stock_suggestions = fetch_and_predict_stocks(suggested_savings)
        st.markdown("---")
        st.subheader("📈 Stock Suggestions Based on Your Budget")
        for ticker, predicted_price in stock_suggestions.items():
            st.write(f"**{ticker}**: Predicted Price for Next Day: ${predicted_price.item():.2f}")
