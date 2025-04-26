import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sqlite3
from ai import train_model, fetch_and_predict_stocks, adjust_non_necessities_for_savings, suggest_budget

# --- Custom CSS for Enhanced Styling ---
st.set_page_config(page_title="Centsible 💸", layout="centered", page_icon="💸")
st.markdown(
    """
    <style>
    /* Overall page style */
    body {
        background-color: #F5F5F5;
    }
    /* White container with soft shadows */
    .main-container {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-bottom:2rem;
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6, p {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    /* Button styling */
    div.stButton>button {
        background-color: #1E90FF;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.7em 1.2em;
        font-weight: bold;
    }
    div.stButton>button:hover {
        background-color: #187bcd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Database Functions ---
def get_db_connection():
    conn = sqlite3.connect('user_data.db')
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT NOT NULL,
                        income REAL)''')
    conn.commit()
    conn.close()

def update_user_income(username, income):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''UPDATE users SET income = ? WHERE username = ?''', (income, username))
    conn.commit()
    conn.close()

def get_user_income(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''SELECT income FROM users WHERE username = ?''', (username,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row['income']
    else:
        return None

# Create table if not exists
create_table()

# --- Session State Initialization ---
if "users" not in st.session_state:
    st.session_state.users = {
        "alice": "password123",
        "bob": "securepass",
    }
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "income" not in st.session_state:
    st.session_state.income = None
if "necessities" not in st.session_state:
    st.session_state.necessities = []
if "non_necessities" not in st.session_state:
    st.session_state.non_necessities = []

# --- Authentication Functions ---
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

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.income = None
    st.session_state.necessities = []
    st.session_state.non_necessities = []
    st.success("You have logged out successfully.")
    st.stop()

# --- Login/Signup UI ---
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>💸 Centsible</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Make Every Cent Count!</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        tabs = st.tabs(["Login", "Sign Up"])
        with tabs[0]:
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username", key="login_username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
                if st.form_submit_button("🚪 Login"):
                    login(username, password)
        with tabs[1]:
            with st.form("signup_form", clear_on_submit=True):
                username = st.text_input("Username", key="signup_username", placeholder="Choose a username")
                password = st.text_input("Password", type="password", key="signup_password", placeholder="Choose a password")
                if st.form_submit_button("📝 Sign Up"):
                    signup(username, password)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Main App: Sidebar Navigation ---
with st.sidebar:
    st.header("💡 Navigation")
    page = st.radio("Select a page", ["Dashboard", "Income", "Expenses", "Budget Analysis"])
    st.button("🚪 Log out", on_click=logout)

# --- Dashboard Page ---
if page == "Dashboard":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>Welcome, {st.session_state.username}!</h2>", unsafe_allow_html=True)
        # Display Income Summary
        if st.session_state.income is None:
            st.info("No income set. Please update your monthly income under the 'Income' section.")
        else:
            st.metric("Monthly Income", f"${st.session_state.income:.2f}")
        # Display Expense Overview
        tot_necessities = sum(val for _, val in st.session_state.necessities)
        tot_non_necessities = sum(val for _, val in st.session_state.non_necessities)
        if st.session_state.necessities or st.session_state.non_necessities:
            st.markdown("### Expense Overview")
            col1, col2 = st.columns(2)
            col1.metric("Total Necessities", f"${tot_necessities:.2f}")
            col2.metric("Total Non-Necessities", f"${tot_non_necessities:.2f}")
        else:
            st.info("No expenses recorded. Please update your expenses under the 'Expenses' section.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Income Page ---
elif page == "Income":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<h2>🧾 Monthly Income</h2>", unsafe_allow_html=True)
        current_income = st.session_state.income or get_user_income(st.session_state.username)
        if current_income:
            st.write(f"**Current monthly income:** ${current_income:.2f}")
        with st.form("income_form", clear_on_submit=True):
            new_income = st.number_input("Enter your monthly income ($)", min_value=0.0, value=current_income or 0.0, step=0.01)
            submitted = st.form_submit_button("Save Income")
            if submitted:
                if new_income > 0:
                    update_user_income(st.session_state.username, new_income)
                    st.session_state.income = new_income
                    st.success(f"Your monthly income of **${new_income:.2f}** has been updated.")
                else:
                    st.error("Income must be greater than zero.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Expenses Page ---
elif page == "Expenses":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<h2>🧾 Monthly Expenses</h2>", unsafe_allow_html=True)
        with st.form("expenses_form", clear_on_submit=True):
            # Necessities Input
            st.markdown("### 📌 Necessities")
            num_necessities = st.number_input("Number of Necessity Categories", min_value=1, value=len(st.session_state.necessities) or 1, step=1, key="num_necessities")
            temp_necessities = []
            for i in range(int(num_necessities)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    name = st.text_input(f"Necessity Category {i+1}", key=f"necessity_name_{i}", placeholder="e.g., Rent")
                with col2:
                    value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"necessity_value_{i}")
                if name:
                    temp_necessities.append((name, value))
            # Non-Necessities Input
            st.markdown("### 🎯 Non-Necessities")
            num_non_necessities = st.number_input("Number of Non-Necessity Categories", min_value=1, value=len(st.session_state.non_necessities) or 1, step=1, key="num_non_necessities")
            temp_non_necessities = []
            for i in range(int(num_non_necessities)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    name = st.text_input(f"Non-Necessity Category {i+1}", key=f"non_necessity_name_{i}", placeholder="e.g., Dining Out")
                with col2:
                    value = st.number_input(f"Value ($)", min_value=0.0, value=0.0, step=0.01, key=f"non_necessity_value_{i}")
                if name:
                    temp_non_necessities.append((name, value))
            if st.form_submit_button("Save Expenses"):
                st.session_state.necessities = temp_necessities
                st.session_state.non_necessities = temp_non_necessities
                st.success("Expenses have been saved.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Budget Analysis Page ---
elif page == "Budget Analysis":
    with st.container():
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<h2>💡 Budget Analysis</h2>", unsafe_allow_html=True)
        if st.session_state.income is None:
            st.error("Please set your monthly income in the 'Income' section first.")
        elif not (st.session_state.necessities or st.session_state.non_necessities):
            st.error("Please add some expenses in the 'Expenses' section first.")
        else:
            with st.spinner("Calculating your budget..."):
                # Combine user expense values
                values = [val for _, val in st.session_state.necessities] + [val for _, val in st.session_state.non_necessities]
                user_input = np.array([values])
                model, scaler_x, scaler_y = train_model(len(values), st.session_state.income)
                user_scaled = scaler_x.transform(user_input)
                user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
                predicted_value = model(user_tensor).item()
                predicted_savings = scaler_y.inverse_transform([[predicted_value]])
                suggested_savings = float(predicted_savings[0][0])
                suggested_spending = st.session_state.income - suggested_savings

                # Refine budget suggestion using the provided helper
                budget_suggestion = suggest_budget(st.session_state.necessities, st.session_state.non_necessities, st.session_state.income)
                suggested_savings = budget_suggestion["suggested_savings"]
                suggested_spending = budget_suggestion["suggested_spending"]
                adjusted_non_necessities = budget_suggestion["adjusted_non_necessities"]

            if suggested_spending < 0:
                st.error("Your expenses exceed your income. Please reduce spending in the 'Expenses' section.")
            else:
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Income", f"${st.session_state.income:.2f}")
                col2.metric("Spending", f"${suggested_spending:.2f}")
                col3.metric("Savings", f"${suggested_savings:.2f}")

                # Progress bar: spending as a percentage of income
                spending_pct = min(suggested_spending / st.session_state.income, 1.0)
                st.markdown("#### Spending as a Portion of Income")
                st.progress(spending_pct)

                st.success(f"✅ You could aim to save **${suggested_savings:.2f}** this month.")
                changes_made = []
                for (orig_cat, orig_val), (adj_cat, adj_val) in zip(st.session_state.non_necessities, adjusted_non_necessities):
                    if orig_val != adj_val:
                        diff = orig_val - adj_val
                        changes_made.append(f"**{adj_cat}**: Adjusted by **${diff:.2f}**")
                if changes_made:
                    st.markdown("### ⚠️ Adjustments Made:")
                    for change in changes_made:
                        st.write(change)

                st.markdown("---")
                st.subheader("💡 Suggested Budget Breakdown")
                for category, value in st.session_state.necessities:
                    st.write(f"- Necessity: **{category}**: ${value:.2f}")
                for category, value in adjusted_non_necessities:
                    st.write(f"- Non-Necessity: **{category}**: ${value:.2f}")
                st.write(f"- 💾 **Savings**: **${suggested_savings:.2f}**")

                # --- Visualizations ---
                st.markdown("## 📊 Visual Breakdown")
                expense_data = {
                    "Necessities": sum(val for _, val in st.session_state.necessities),
                    "Non-Necessities": sum(val for _, val in adjusted_non_necessities),
                    "Savings": suggested_savings
                }
                # Pie Chart using Matplotlib
                fig1, ax1 = plt.subplots()
                ax1.pie(expense_data.values(), labels=expense_data.keys(), autopct='%1.1f%%', startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1)

                # Bar Chart for a detailed breakdown
                categories = [name for name, _ in st.session_state.necessities] + [name for name, _ in adjusted_non_necessities] + ["Savings"]
                amounts = [val for _, val in st.session_state.necessities] + [val for _, val in adjusted_non_necessities] + [suggested_savings]
                df_bar = pd.DataFrame({"Category": categories, "Amount ($)": amounts})
                st.bar_chart(df_bar.set_index("Category"))

                # --- Stock Suggestions ---
                stock_suggestions = fetch_and_predict_stocks(suggested_savings)
                st.markdown("---")
                st.subheader("📈 Stock Suggestions Based on Your Budget")
                for ticker, predicted_price in stock_suggestions.items():
                    try:
                        predicted_price = float(predicted_price)
                        st.write(f"**{ticker}**: Predicted Next-Day Price: **${predicted_price:.2f}**")
                    except ValueError:
                        st.write(f"**{ticker}**: Predicted Next-Day Price: **Invalid Data**")
        st.markdown("</div>", unsafe_allow_html=True)
