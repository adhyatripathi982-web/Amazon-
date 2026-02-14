import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Amazon Analytics Dashboard",
    layout="wide",
    page_icon="📦"
)

# ------------------------------------------------
# CUSTOM AMAZON STYLING
# ------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #F6F6F6;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #FF9900;
}
h1, h2, h3 {
    color: #232F3E;
}
.sidebar .sidebar-content {
    background-color: #232F3E;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER WITH LOGO
# ------------------------------------------------
col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=120)

with col2:
    st.title("Amazon Fulfillment & Retention Intelligence Dashboard")
    st.caption("Data-driven insights for operational excellence")

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.header("📊 Dataset Options")

dataset_option = st.sidebar.radio(
    "Choose Dataset Source",
    ["Generate Sample Dataset", "Upload CSV/Excel"]
)

df = None

# ------------------------------------------------
# DATA LOADING
# ------------------------------------------------
if dataset_option == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("Dataset Loaded Successfully")
        except Exception as e:
            st.sidebar.error("Error loading dataset")
            st.sidebar.exception(e)

else:
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "Order_ID": range(1, n+1),
        "Customer_ID": np.random.randint(1000, 1100, n),
        "Order_Date": pd.date_range(start='2023-01-01', periods=n, freq='D'),
        "Delivery_Date": pd.date_range(start='2023-01-03', periods=n, freq='D') +
                         pd.to_timedelta(np.random.randint(1, 7, n), unit='D'),
        "Order_Accuracy": np.random.choice([1, 0], n, p=[0.95, 0.05]),
        "Stock_Level": np.random.randint(50, 500, n),
        "Inventory_Age_Days": np.random.randint(1, 100, n),
        "Shipping_Cost": np.random.randint(5, 50, n),
        "Return_Reason": np.random.choice(
            ["Damaged", "Late Delivery", "Not Needed", "Wrong Item"], n
        ),
        "Purchase_Frequency": np.random.randint(1, 10, n),
        "Monetary_Value": np.random.randint(20, 500, n),
    })

# Stop if no dataset
if df is None:
    st.warning("Please upload or generate dataset.")
    st.stop()

# ------------------------------------------------
# DATA CLEANING
# ------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
df["Lead_Time_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days
df["RFM_Score"] = df["Purchase_Frequency"] * df["Monetary_Value"]
df["Normalized_Shipping_Cost"] = MinMaxScaler().fit_transform(df[["Shipping_Cost"]])

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input(
    "Filter by Order Date",
    [df["Order_Date"].min(), df["Order_Date"].max()]
)

return_filter = st.sidebar.multiselect(
    "Filter by Return Reason",
    options=df["Return_Reason"].unique(),
    default=df["Return_Reason"].unique()
)

df = df[
    (df["Order_Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order_Date"] <= pd.to_datetime(date_range[1])) &
    (df["Return_Reason"].isin(return_filter))
]

# ------------------------------------------------
# KPI SECTION
# ------------------------------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", len(df))

with col2:
    st.metric("Avg Lead Time", f"{df['Lead_Time_Days'].mean():.1f} Days")

with col3:
    st.metric("Order Accuracy", f"{df['Order_Accuracy'].mean()*100:.1f}%")

with col4:
    st.metric("Avg Revenue per Order", f"${df['Monetary_Value'].mean():.2f}")

# ------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------
st.subheader("📊 Operational Analytics")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["Lead_Time_Days"], kde=True, color="#FF9900", ax=ax)
    ax.set_title("Lead Time Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="Inventory_Age_Days",
        y="Stock_Level",
        data=df,
        color="#232F3E",
        ax=ax
    )
    ax.set_title("Inventory Age vs Stock")
    st.pyplot(fig)

st.subheader("📦 Returns & Customer Behavior")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df["Return_Reason"].value_counts().plot(
        kind="bar",
        color="#FF9900",
        ax=ax
    )
    ax.set_title("Return Reason Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(df["RFM_Score"], color="#232F3E", ax=ax)
    ax.set_title("Customer RFM Score Distribution")
    st.pyplot(fig)

# ------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ------------------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------------------
st.subheader("💡 Strategic Insights")

st.info("""
• Reduce lead time variability to improve Prime-level experience  
• Optimize aging inventory to reduce holding cost  
• Address dominant return reasons to reduce reverse logistics  
• Identify high RFM customers for targeted retention campaigns  
• Improve operational efficiency through predictive analytics  
""")
