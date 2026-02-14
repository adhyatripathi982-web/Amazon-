import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# ---------------------------
# Title
# ---------------------------
st.set_page_config(page_title="Amazon Dashboard", layout="wide")
st.title("Amazon Fulfillment & Viewer Retention Dashboard")

# ---------------------------
# Dataset Upload or Generate
# ---------------------------
st.sidebar.header("Dataset Options")
dataset_option = st.sidebar.selectbox(
    "Choose Dataset Option",
    ["Upload CSV/Excel", "Generate Sample Dataset"]
)

df = None  # 🔥 IMPORTANT: Initialize df

if dataset_option == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("Dataset loaded successfully!")

        except Exception as e:
            st.error("Error loading file")
            st.exception(e)

else:
    st.info("Generating sample dataset...")
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "Order_ID": range(1, n+1),
        "Customer_ID": np.random.randint(1000, 1100, n),
        "Order_Date": pd.date_range(start='2023-01-01', periods=n, freq='D'),
        "Delivery_Date": pd.date_range(start='2023-01-03', periods=n, freq='D') +
                         pd.to_timedelta(np.random.randint(1, 7, n), unit='D'),
        "Order_Accuracy": np.random.choice([1, 0], n, p=[0.95, 0.05]),
        "Defect_Rate": np.random.rand(n),
        "Stock_Level": np.random.randint(50, 500, n),
        "Inventory_Age_Days": np.random.randint(1, 100, n),
        "Shipping_Cost": np.random.randint(5, 50, n),
        "FBA_Fees": np.random.randint(2, 20, n),
        "3PL_Cost": np.random.randint(3, 25, n),
        "On_Time_Delivery": np.random.choice([1, 0], n, p=[0.9, 0.1]),
        "Return_Flag": np.random.choice([1, 0], n, p=[0.1, 0.9]),
        "Return_Reason": np.random.choice(
            ["Damaged", "Late Delivery", "Not Needed", "Wrong Item"], n
        ),
        "Purchase_Frequency": np.random.randint(1, 10, n),
        "Monetary_Value": np.random.randint(20, 500, n),
        "Subscription_Flag": np.random.choice([1, 0], n, p=[0.3, 0.7])
    })

    st.success("Sample dataset generated!")

# ---------------------------
# STOP if no dataset
# ---------------------------
if df is None:
    st.warning("Please upload a dataset or generate a sample dataset.")
    st.stop()

# ---------------------------
# Preview Dataset
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Data Cleaning
# ---------------------------
st.subheader("Data Cleaning")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    st.write("Missing values handled for numeric columns.")

# ---------------------------
# Feature Engineering
# ---------------------------
st.subheader("Feature Engineering")

if "Order_Date" in df.columns and "Delivery_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')
    df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors='coerce')
    df["Lead_Time_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days

if "Purchase_Frequency" in df.columns and "Monetary_Value" in df.columns:
    df["RFM_Score"] = df["Purchase_Frequency"] * df["Monetary_Value"]

if "Shipping_Cost" in df.columns:
    df["Normalized_Shipping_Cost"] = MinMaxScaler().fit_transform(
        df[["Shipping_Cost"]]
    )

st.success("Feature engineering completed.")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

if "Lead_Time_Days" in df.columns:
    st.write("### Lead Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Lead_Time_Days"].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

if "Order_Accuracy" in df.columns:
    st.write("### Order Accuracy Rate")
    accuracy_rate = df["Order_Accuracy"].mean()
    st.metric("Order Accuracy", f"{accuracy_rate*100:.2f}%")

if {"Inventory_Age_Days", "Stock_Level"}.issubset(df.columns):
    st.write("### Inventory Age vs Stock Level")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Inventory_Age_Days", y="Stock_Level", data=df, ax=ax)
    st.pyplot(fig)

if {"Shipping_Cost", "FBA_Fees", "3PL_Cost"}.issubset(df.columns):
    st.write("### Shipping Costs")
    fig, ax = plt.subplots()
    df[["Shipping_Cost", "FBA_Fees", "3PL_Cost"]].hist(ax=ax)
    st.pyplot(fig)

if "Return_Reason" in df.columns:
    st.write("### Return Reasons Distribution")
    fig, ax = plt.subplots()
    df["Return_Reason"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ---------------------------
# Business Insights
# ---------------------------
st.subheader("Business Insights")
st.write("""
• Monitor delivery lead time to improve logistics efficiency.  
• High order accuracy improves customer satisfaction.  
• Control inventory aging to reduce holding costs.  
• Optimize shipping cost mix (FBA vs 3PL).  
• Reduce returns by analyzing top return reasons.  
• Use RFM and CLV metrics to identify high-value customers.  
""")
