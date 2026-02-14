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
st.title("Amazon Fulfillment & Viewer Retention Dashboard")

# ---------------------------
# Dataset Upload or Generate
# ---------------------------
st.sidebar.header("Dataset Options")
dataset_option = st.sidebar.selectbox("Choose Dataset Option", ["Upload CSV/Excel", "Generate Sample Dataset"])

if dataset_option == "Upload CSV/Excel":
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("Dataset loaded successfully!")
else:
    st.info("Generating sample dataset...")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Order_ID": range(1, n+1),
        "Customer_ID": np.random.randint(1000, 1100, n),
        "Order_Date": pd.date_range(start='2023-01-01', periods=n, freq='D'),
        "Delivery_Date": pd.date_range(start='2023-01-03', periods=n, freq='D') + pd.to_timedelta(np.random.randint(1, 7, n), unit='D'),
        "Order_Accuracy": np.random.choice([1, 0], n, p=[0.95, 0.05]),
        "Defect_Rate": np.random.rand(n),
        "Stock_Level": np.random.randint(50, 500, n),
        "Inventory_Age_Days": np.random.randint(1, 100, n),
        "Shipping_Cost": np.random.randint(5, 50, n),
        "FBA_Fees": np.random.randint(2, 20, n),
        "3PL_Cost": np.random.randint(3, 25, n),
        "On_Time_Delivery": np.random.choice([1, 0], n, p=[0.9, 0.1]),
        "Return_Flag": np.random.choice([1, 0], n, p=[0.1, 0.9]),
        "Return_Reason": np.random.choice(["Damaged", "Late Delivery", "Not Needed", "Wrong Item"], n),
        "Purchase_Frequency": np.random.randint(1, 10, n),
        "Monetary_Value": np.random.randint(20, 500, n),
        "Subscription_Flag": np.random.choice([1, 0], n, p=[0.3, 0.7])
    })
    st.success("Sample dataset generated!")

# ---------------------------
# Preview Dataset
# ---------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Data Cleaning
# ---------------------------
st.subheader("Data Cleaning")
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
st.write("Missing values handled for numeric columns.")

# ---------------------------
# Feature Engineering
# ---------------------------
st.subheader("Feature Engineering")
df["Lead_Time_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days
df["RFM_Score"] = df["Purchase_Frequency"] * df["Monetary_Value"]
df["Normalized_Shipping_Cost"] = MinMaxScaler().fit_transform(df[["Shipping_Cost"]])
st.write("Features 'Lead_Time_Days', 'RFM_Score', 'Normalized_Shipping_Cost' created.")

# ---------------------------
# Visualizations
# ---------------------------
st.subheader("Visualizations")

# Lead Time Analysis
st.write("### Lead Time Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Lead_Time_Days"], kde=True, ax=ax)
st.pyplot(fig)

# Order Accuracy
st.write("### Order Accuracy Rate")
accuracy_rate = df["Order_Accuracy"].mean()
st.metric("Order Accuracy", f"{accuracy_rate*100:.2f}%")

# Inventory Health
st.write("### Inventory Age vs Stock Level")
fig, ax = plt.subplots()
sns.scatterplot(x="Inventory_Age_Days", y="Stock_Level", data=df, ax=ax)
st.pyplot(fig)

# Shipping Cost Breakdown
st.write("### Shipping Costs")
fig, ax = plt.subplots()
df[["Shipping_Cost","FBA_Fees","3PL_Cost"]].hist(ax=ax)
st.pyplot(fig)

# Returns Analysis
st.write("### Return Reasons Distribution")
fig, ax = plt.subplots()
df["Return_Reason"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# Cohort Analysis (Simple Monthly Cohort)
st.write("### Cohort Analysis - Orders per Month")
df['Order_Month'] = df['Order_Date'].dt.to_period('M')
cohort_data = df.groupby('Order_Month')['Customer_ID'].nunique()
st.bar_chart(cohort_data)

# Repeat Purchase Rate
st.write("### Repeat Purchase Rate")
repeat_customers = df[df["Purchase_Frequency"] > 1]["Customer_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
rpr = repeat_customers / total_customers
st.metric("Repeat Purchase Rate", f"{rpr*100:.2f}%")

# CLV (Simplified)
st.write("### Customer Lifetime Value (CLV)")
clv = df.groupby("Customer_ID")["Monetary_Value"].sum().mean()
st.metric("Average CLV", f"${clv:.2f}")

# ---------------------------
# Business Insights
# ---------------------------
st.subheader("Business Insights")
st.write("""
1. **Lead Time Analysis:** Average delivery lead time can be monitored to optimize logistics.
2. **Order Accuracy:** High accuracy reduces returns and improves customer satisfaction.
3. **Inventory Health:** Monitor aging inventory to avoid stockouts or excess holding costs.
4. **Shipping Costs:** Compare FBA vs 3PL to optimize margins.
5. **Returns:** Understand common return reasons to reduce defect rates.
6. **Retention:** Use RFM, cohort, repeat purchase, and CLV to identify loyal customers.
7. **Subscription Performance:** Monitor subscription adoption and opt-out rates.
""")
