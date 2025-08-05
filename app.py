import streamlit as st
import pandas as pd
from visuals.plots import plot_price_vs_sentiment, plot_return_boxplot, plot_corr_heatmap
from visuals.insights import generate_sentiment_summary, generate_observations
from models.model import run_logistic_model
from visuals.plots import plot_price_with_moving_averages
from visuals.plots import plot_return_histogram
from visuals.plots import (
    plot_price_vs_sentiment,
    plot_return_boxplot,
    plot_corr_heatmap,
    plot_feature_importance,
    plot_volatility_trendline
)




# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/featured_data.csv", parse_dates=["date"])

df = load_data()

# Page config
st.set_page_config(page_title="Bitcoin Sentiment Analysis", layout="wide")
st.title("📊 Bitcoin Fear & Greed Sentiment Dashboard")
st.markdown("Built by Tanvi Sundarkar • Inspired by [Gaies et al., 2023]")

# Sidebar Filters
st.sidebar.header("🔎 Filter Data")
start_date = st.sidebar.date_input("Start Date", df['date'].min().date())
end_date = st.sidebar.date_input("End Date", df['date'].max().date())

df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Price & Sentiment",
    "📊 Return vs Sentiment",
    "🧠 Correlations",
    "📌 Insights",
    "📈 Prediction"
])



with tab1:
    st.subheader("Bitcoin Closing Price vs Fear & Greed Index")
    st.plotly_chart(plot_price_vs_sentiment(df_filtered), use_container_width=True)
    st.markdown("### 📘 What this chart shows")
    st.markdown("""
        - The **blue line** represents Bitcoin’s daily closing price.
        - The **orange line** shows the Fear & Greed Index (0-100 scale).
        - Periods of high **greed** (FGI > 60) often precede **price drops**, while high **fear** (FGI < 30) may precede **price increases**.
        - This inverse relationship suggests that **sentiment can be a contrarian indicator**.
        """)

    st.subheader("Bitcoin Price with Moving Averages")
    st.plotly_chart(plot_price_with_moving_averages(df_filtered), use_container_width=True)

with tab2:
    st.subheader("Distribution of Daily Returns by Sentiment")
    st.plotly_chart(plot_return_boxplot(df_filtered), use_container_width=True)
    st.markdown("### 📘 What this chart shows")
    st.markdown("""
        - This boxplot shows how **daily returns** vary based on sentiment categories like *Fear*, *Neutral*, and *Greed*.
        - You can observe **median returns**, **spread (volatility)**, and potential **outliers** for each sentiment group.
        - For example, the *Fear* group might show **higher positive spikes**, while *Greed* often has **lower or negative returns**, hinting at **risk of corrections** during overly optimistic periods.
        """)

    st.subheader("Histogram of Daily Returns")
    st.plotly_chart(plot_return_histogram(df_filtered), use_container_width=True, key="return_hist")

with tab3:
    st.subheader("Feature Correlation Heatmap")
    st.pyplot(plot_corr_heatmap(df_filtered))
    st.markdown("### 📘 What this chart shows")
    st.markdown("""
        - This heatmap shows **pairwise correlations** between numerical features such as:
            - `fgi_value`, `daily_return`, `volatility`, etc.
        - Correlation values range from **-1 to 1**:
            - **1** = perfect positive correlation (both rise together)
            - **-1** = perfect negative correlation (one rises, other falls)
            - **0** = no correlation
        - Look for strong correlations like:
            - `fgi_value` and `fgi_value_lag1` (high positive)
            - `daily_return` and `volatility` (may be low or negative)
        - These help in **feature selection** and **understanding market behavior**.
        """)
with tab4:
    st.subheader("🧠 Behavioral Insights from Sentiment States")

    # Summary table
    st.markdown("### 📊 Summary Table")
    st.dataframe(generate_sentiment_summary(df_filtered), use_container_width=True)

    st.markdown("### 📘 What this table shows")
    st.markdown("""
    - This table groups the data by **sentiment state** (e.g., *Fear*, *Greed*, etc.).
    - For each state, we calculate:
        - Average **daily return**
        - Average **volatility**
        - Number of days in that sentiment state
    - You can compare how **returns and volatility differ** across different sentiment levels.
    - For example:
        - During **Extreme Greed**, average returns may be higher.
        - During **Fear**, volatility might spike.
    """)

    # Key observations
    st.markdown("### 📝 Key Observations")
    for insight in generate_observations(df_filtered):
        st.success(insight)

    st.markdown("### 📘 How to interpret the observations")
    st.markdown("""
    - These insights summarize **patterns between sentiment and market behavior**.
    - They help you understand **how sentiment affects price direction and volatility**.
    - Use them to build intuition for trading decisions or model features.
    """)

with tab5:
    st.subheader("📈 Predicting Price Direction using Sentiment")

    # Run model
    acc, prediction, coefs = run_logistic_model(df_filtered)

    # Accuracy metric
    st.metric("Model Accuracy", f"{acc * 100:.2f}%")
    st.caption("📌 This tells how often the model correctly predicted whether BTC will go up or down the next day.")

    # Next-day prediction
    st.subheader("🔮 Tomorrow's BTC Price Direction Prediction")
    direction = "⬆️ Up" if prediction == 1 else "⬇️ Down"
    st.success(f"Predicted Direction: **{direction}**")
    st.caption("📌 Prediction based on today’s sentiment score, daily return, and volatility.")

    # Feature Importance
    st.subheader("📊 Feature Importance (Logistic Coefficients)")
    st.plotly_chart(plot_feature_importance(coefs), use_container_width=True)
    st.caption("📌 Features with higher absolute coefficients have more influence on the prediction.")
    st.markdown("### 📘 What do these coefficients mean?")
    st.markdown("""
    - These values show how much **each feature influences the prediction**.
    - Positive values → Feature increases the chance Bitcoin will go **up** tomorrow.
    - Negative values → Feature increases the chance Bitcoin will go **down** tomorrow.
    - Bigger the absolute value → stronger the impact.

    **Examples:**
    - If `daily_return` has a strong positive value → recent gains hint at more gains.
    - If `sentiment_encoded` is negative → fear in market may signal losses.
    """)

    # Volatility trendline
    st.subheader("📉 Volatility Trend (7-Day Rolling)")
    st.plotly_chart(plot_volatility_trendline(df_filtered), use_container_width=True)
    st.caption("📌 Volatility indicates how much BTC price fluctuates over time.")
    st.markdown("### 📘 Why is volatility important?")
    st.markdown("""
    - Higher volatility means more price movement, which could mean **higher risk** or **market stress**.
    - It helps traders gauge market uncertainty.
    - A sudden spike in volatility often comes before major moves.
    """)

    # CSV Download
    st.subheader("💾 Download Filtered Dataset")
    st.download_button(
        label="Download CSV",
        data=df_filtered.to_csv(index=False).encode(),
        file_name="filtered_bitcoin_sentiment.csv",
        mime="text/csv"
    )
    st.caption("📌 Export your filtered dataset for further offline analysis or Excel visualization.")

    # Explain accuracy
    st.markdown("### 📘 What does model accuracy mean?")
    st.markdown("""
    - The model predicts whether the **next day’s BTC return** will be positive or negative.
    - Accuracy is the percentage of correct predictions the model made.
    - For example, **80% accuracy** means it correctly predicted 8 out of 10 recent cases.
    """)

with st.expander("ℹ️ How to interpret these charts?"):
    st.markdown("""
    - **Price vs Sentiment**: Understand emotional phases of the market.
    - **Return Boxplot**: See how emotion affects volatility.
    - **Heatmap**: Spot relationships between features.
    - **Prediction**: Estimate next-day price direction from sentiment.
    """)

# Footer
st.markdown("---")
st.caption("Data sources: [Alternative.me](https://alternative.me), [Yahoo Finance](https://finance.yahoo.com/)")
