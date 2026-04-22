
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Finance Digital Twin ", layout="wide")

st.title("💰 AI Finance Digital Twin")

# Load data
try:
    df = pd.read_csv("data.csv")
except:
    df = pd.DataFrame(columns=["date","category","amount","mode"])

# Sidebar input
st.sidebar.header("Add Transaction")
date = st.sidebar.date_input("Date")
category = st.sidebar.selectbox("Category", ["Food","Travel","Shopping","Bills","Other"])
amount = st.sidebar.number_input("Amount", min_value=0)
mode = st.sidebar.selectbox("Mode", ["UPI","Cash","Card"])

if st.sidebar.button("Add"):
    new_row = {"date":str(date),"category":category,"amount":amount,"mode":mode}
    df = pd.concat([df, pd.DataFrame([new_row])])
    df.to_csv("data.csv", index=False)
    st.sidebar.success("Added!")

# Convert date
if len(df) > 0:
    df["date"] = pd.to_datetime(df["date"])

# Metrics
if len(df) > 0:
    total_spent = df["amount"].sum()
    avg_spent = df["amount"].mean()
    upi_spent = df[df["mode"]=="UPI"]["amount"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spent", f"₹{total_spent:.0f}")
    col2.metric("Average Expense", f"₹{avg_spent:.0f}")
    col3.metric("UPI Usage", f"₹{upi_spent:.0f}")

# Charts
st.subheader("📊 Expense by Category")
if len(df) > 0:
    st.bar_chart(df.groupby("category")["amount"].sum())

st.subheader("📈 Daily Spending Trend")
if len(df) > 0:
    df_sorted = df.sort_values("date")
    st.line_chart(df_sorted.set_index("date")["amount"])

# Weekly Report
st.subheader("📅 Weekly Report")
if len(df) > 0:
    df["week"] = df["date"].dt.isocalendar().week
    weekly = df.groupby("week")["amount"].sum()
    st.write(weekly)

# ML Prediction
st.subheader("🤖 Expense Prediction")
if len(df) > 5:
    df["day"] = df["date"].dt.day
    X = df[["day"]]
    y = df["amount"]

    model = LinearRegression()
    model.fit(X, y)

    future_day = st.slider("Select future day", 1, 31, 15)
    pred = model.predict([[future_day]])

    st.success(f"Predicted expense: ₹{pred[0]:.2f}")

# Smart Insight
st.subheader("🧠 Smart Insight")
if len(df) > 0:
    top_category = df.groupby("category")["amount"].sum().idxmax()
    st.warning(f"You spend most on {top_category}. Try reducing it!")
