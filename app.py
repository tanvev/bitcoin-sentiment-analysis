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
from models.model import run_logistic_model
from models.prediction import load_or_create_prediction
import os
import pandas as pd
from datetime import datetime
import yfinance as yf







# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/featured_data.csv", parse_dates=["date"])

df = load_data()

# 🛠️ Ensure required columns exist
if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
# Add derived columns if missing
if 'daily_return' not in df.columns:
    df['daily_return'] = df['close'].pct_change()

if 'volatility' not in df.columns:
    df['volatility'] = df['daily_return'].rolling(window=7).std()

if 'sentiment_encoded' not in df.columns:
    # Simple encoding: Fear = 0, Neutral = 1, Greed = 2 (modify if you use other states)
    sentiment_map = {'Fear': 0, 'Neutral': 1, 'Greed': 2}
    df['sentiment_encoded'] = df['fgi_sentiment'].map(sentiment_map)

if 'target' not in df.columns:
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Remove NaNs introduced by pct_change and rolling
df.dropna(inplace=True)


# Page config
st.set_page_config(page_title="Bitcoin Sentiment Analysis", layout="wide")
st.title("📊 Bitcoin Fear & Greed Sentiment Dashboard")
st.markdown("Built by Tanvi Sundarkar • Inspired by [Gaies et al., 2023]")

# Sidebar Filters
st.sidebar.header("🔎 Filter Data")
start_date = st.sidebar.date_input("Start Date", df['date'].min().date())
end_date = st.sidebar.date_input("End Date", df['date'].max().date())
csv_path = "model.predictions.csv"
if os.path.exists(csv_path):
    pred_df = pd.read_csv(csv_path)
    pred_df = pred_df[pred_df["is_correct"] != "N/A"]

    if not pred_df.empty:
        accuracy_over_time = pred_df["is_correct"].astype(int).mean()
        st.metric("📊 Historical Accuracy", f"{accuracy_over_time * 100:.2f}%")

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

    # Generate or load prediction
    pred, accuracy, coefs = load_or_create_prediction(run_logistic_model, df)

    today = datetime.now().strftime("%Y-%m-%d")
    st.markdown(f"**Prediction for {today}:** BTC will **{'rise 📈' if pred == 1 else 'fall 📉'}** tomorrow.")
    st.markdown(f"**Model Accuracy on Training Data:** {accuracy:.2%}")

    # Feature importance table
    if coefs is not None:
        st.subheader("🔍 Feature Importance (Coefficients)")
        st.dataframe(coefs)

    # Feature importance bar chart


    st.subheader("📊 Feature Influence")
    st.plotly_chart(plot_feature_importance(coefs), use_container_width=True)

    # Live BTC price using yfinance
    btc = yf.Ticker("BTC-USD")
    try:
        live_price = btc.history(period="1d")["Close"].iloc[-1]
        st.metric("💰 Current BTC Price", f"${live_price:,.2f}")
    except:
        st.warning("⚠️ Unable to fetch live BTC price.")

    # Volatility trend
    st.subheader("📉 Volatility Trend (7-Day Rolling)")
    st.plotly_chart(plot_volatility_trendline(df_filtered), use_container_width=True)

    # Historical prediction tracker
    st.subheader("📅 Historical Prediction Accuracy Tracker")
    csv_path = "data/predictions.csv"
    if os.path.exists(csv_path):
        hist_df = pd.read_csv(csv_path)
        if "date" in hist_df.columns and "accuracy" in hist_df.columns and "predicted_direction" in hist_df.columns:
            hist_df["date"] = pd.to_datetime(hist_df["date"])
            hist_df = hist_df.set_index("date")

            st.line_chart(hist_df["predicted_direction"])
            st.caption("Shows the predicted direction over time.")

            acc = hist_df["accuracy"].dropna().mean()
            st.metric("📊 Average Historical Accuracy", f"{acc:.2%}")
        else:
            st.warning("❌ Missing required columns in predictions.csv")
    else:
        st.warning("📁 predictions.csv not found. Predictions will be stored here after first run.")

    # Dataset download
    st.subheader("💾 Download Filtered Dataset")
    st.download_button(
        label="Download CSV",
        data=df_filtered.to_csv(index=False).encode(),
        file_name="filtered_bitcoin_sentiment.csv",
        mime="text/csv"
    )

    # Interpretations
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
