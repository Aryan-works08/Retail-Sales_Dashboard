import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.express as px

# ================== Load Data ==================
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/pooja/Downloads/retail_sales_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])  # Capital D
    return df

df = load_data()

# ================== Sidebar Filters ==================
st.sidebar.title("Filters")
category_filter = st.sidebar.multiselect("Select Product Category", df["Product Category"].unique())
gender_filter = st.sidebar.multiselect("Select Gender", df["Gender"].unique())

# Apply filters
filtered_df = df.copy()
if category_filter:
    filtered_df = filtered_df[filtered_df["Product Category"].isin(category_filter)]
if gender_filter:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]

# ================== KPIs ==================
st.title("📊 Retail Sales Forecasting Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"₹{filtered_df['Total Amount'].sum():,.0f}")
col2.metric("Unique Customers", filtered_df["Customer ID"].nunique())
col3.metric("Avg Order Value", f"₹{filtered_df['Total Amount'].mean():,.0f}")

# ================== EDA: Sales Trends ==================
st.subheader("Sales Trends Over Time")
sales_over_time = filtered_df.groupby("Date")["Total Amount"].sum().reset_index()
fig_time = px.line(sales_over_time, x="Date", y="Total Amount",
                   title="Revenue Over Time", markers=True)
st.plotly_chart(fig_time, use_container_width=True)

# ================== Forecast ==================
st.subheader("Sales Forecast")

df_prophet = filtered_df.groupby("Date")["Total Amount"].sum().reset_index()
df_prophet.columns = ["ds", "y"]

if not df_prophet.empty:
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x="ds", y="yhat", title="30-Day Sales Forecast")
    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                             mode="lines", line=dict(width=0), showlegend=False)
    fig_forecast.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                             mode="lines", line=dict(width=0), fill="tonexty",
                             name="Confidence Interval")
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("⚠️ Not enough data for forecasting.")

# ================== Extra Visuals ==================
st.subheader("Sales by Product Category")
category_sales = filtered_df.groupby("Product Category")["Total Amount"].sum().reset_index()
fig_cat = px.bar(category_sales, x="Product Category", y="Total Amount",
                 title="Revenue by Product Category", text_auto=True, color="Product Category")
st.plotly_chart(fig_cat, use_container_width=True)

st.subheader("Sales by Gender")
gender_sales = filtered_df.groupby("Gender")["Total Amount"].sum().reset_index()
fig_gender = px.pie(gender_sales, names="Gender", values="Total Amount",
                    title="Revenue Contribution by Gender", hole=0.4)
st.plotly_chart(fig_gender, use_container_width=True)

st.subheader("Top 10 Customers")
top_customers = filtered_df.groupby("Customer ID")["Total Amount"].sum().nlargest(10).reset_index()
fig_top = px.bar(top_customers, x="Total Amount", y="Customer ID", orientation="h",
                 title="Top 10 Customers by Revenue", text_auto=True, color="Total Amount")
st.plotly_chart(fig_top, use_container_width=True)

st.subheader("Monthly Revenue Trend")
filtered_df["Month"] = filtered_df["Date"].dt.to_period("M")
monthly_sales = filtered_df.groupby("Month")["Total Amount"].sum().reset_index()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)  # Convert period to string for plotting
fig_month = px.area(monthly_sales, x="Month", y="Total Amount",
                    title="Monthly Revenue Trend", markers=True)
st.plotly_chart(fig_month, use_container_width=True)
