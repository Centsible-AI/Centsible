import streamlit as st
import numpy as np
import torch
from ai import train_model, fetch_and_predict_stocks, adjust_non_necessities_for_savings, suggest_budget

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

# Expense categories - Necessities and Non-Necessities
st.write("Enter your custom expense categories:")

# Input for Necessities
st.subheader("Necessities")
necessities = []
num_necessities = st.number_input("Number of Necessity Categories", min_value=1, value=1, step=1)
for i in range(int(num_necessities)):
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input(f"Necessity Category {i+1} Name", key=f"necessity_name_{i}")
    with col2:
        value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"necessity_value_{i}")
    if name:
        necessities.append((name, value))

# Input for Non-Necessities
st.subheader("Non-Necessities")
non_necessities = []
num_non_necessities = st.number_input("Number of Non-Necessity Categories", min_value=1, value=1, step=1)
for i in range(int(num_non_necessities)):
    col1, col2 = st.columns([2, 1])
    with col1:
        name = st.text_input(f"Non-Necessity Category {i+1} Name", key=f"non_necessity_name_{i}")
    with col2:
        value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"non_necessity_value_{i}")
    if name:
        non_necessities.append((name, value))

# Show entered categories
if necessities or non_necessities:
    st.write("Your entered expense categories:")
    for category, value in necessities:
        st.write(f"- Necessity: **{category}**: ${value:.2f}")
    for category, value in non_necessities:
        st.write(f"- Non-Necessity: **{category}**: ${value:.2f}")

# Suggest budget
if st.button("Suggest Budget"):
    if not (necessities or non_necessities):
        st.error("Please add categories before suggesting a budget.")
    else:
        # Gather the expenses and create input array
        values = [val for _, val in necessities] + [val for _, val in non_necessities]
        user_input = np.array([values])

        # Get the model and scalers
        model, scaler_x, scaler_y = train_model(len(necessities) + len(non_necessities), monthly_income)
        
        # Scale and transform the user input
        user_scaled = scaler_x.transform(user_input)
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
        
        # Get predicted savings
        predicted_value = model(user_tensor).item()
        predicted_savings = scaler_y.inverse_transform([[predicted_value]])
        suggested_savings = float(predicted_savings[0][0])

        # Ensure the suggested savings and spending are reasonable
        suggested_spending = monthly_income - suggested_savings

        # Use the new adjusted budget suggestion function
        budget_suggestion = suggest_budget(necessities, non_necessities, monthly_income)
        suggested_savings = budget_suggestion["suggested_savings"]
        suggested_spending = budget_suggestion["suggested_spending"]
        adjusted_non_necessities = budget_suggestion["adjusted_non_necessities"]

        # Display results
        if suggested_spending < 0:
            st.error("Your expenses exceed your income. Please adjust the values.")
        else:
            st.success(f"✅ You could aim to save **${suggested_savings:.2f}** this month.")

        # Show the changes made to non-necessities
            changes_made = []
            for (category, original_value), (adjusted_category, adjusted_value) in zip(non_necessities, adjusted_non_necessities):
                if original_value != adjusted_value:
                    change = original_value - adjusted_value
                    changes_made.append(f"**{adjusted_category}**: Adjusted by **${change:.2f}**")

            if changes_made:
                st.write("⚠️ Key Updates:")
                for change in changes_made:
                    st.write(change)

            st.markdown("---")
            st.subheader("💡 Suggested Budget Breakdown")
            for category, value in necessities:
                st.write(f"- Necessity: **{category}**: ${value:.2f}")
            for category, value in adjusted_non_necessities:
                st.write(f"- Non-Necessity: **{category}**: ${value:.2f}")
            st.write(f"- 💾 **Savings**: **${suggested_savings:.2f}**")

            # Fetch and display stock suggestions
            stock_suggestions = fetch_and_predict_stocks(suggested_savings)
            st.markdown("---")
            st.subheader("📈 Stock Suggestions Based on Your Budget")
            for ticker, predicted_price in stock_suggestions.items():
                st.write(f"**{ticker}**: Predicted Price for Next Day: ${predicted_price:.2f}")
