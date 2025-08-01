import streamlit as st
import pandas as pd
from visuals.plots import plot_price_vs_sentiment, plot_return_boxplot, plot_corr_heatmap
from visuals.insights import generate_sentiment_summary, generate_observations
from models.model import run_logistic_model



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

with tab2:
    st.subheader("Distribution of Daily Returns by Sentiment")
    st.plotly_chart(plot_return_boxplot(df_filtered), use_container_width=True)

with tab3:
    st.subheader("Feature Correlation Heatmap")
    st.pyplot(plot_corr_heatmap(df_filtered))
with tab4:
    st.subheader("🧠 Behavioral Insights from Sentiment States")

    # Summary table
    st.markdown("### 📊 Summary Table")
    st.dataframe(generate_sentiment_summary(df_filtered), use_container_width=True)

    # Key observations
    st.markdown("### 📝 Key Observations")
    for insight in generate_observations(df_filtered):
        st.success(insight)

with tab5:
    st.subheader("📈 Predicting Price Direction using Sentiment")

    acc, coefs = run_logistic_model(df_filtered)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    st.markdown("### 🔍 Coefficient Interpretation")
    st.dataframe(coefs, use_container_width=True)

    st.info("Higher coefficient = stronger influence on predicting whether BTC will go up next day.")


# Footer
st.markdown("---")
st.caption("Data sources: [Alternative.me](https://alternative.me), [Yahoo Finance](https://finance.yahoo.com/)")
