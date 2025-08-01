import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Line plot: BTC price + FGI over time
def plot_price_vs_sentiment(df):
    fig = px.line(df, x='date', y=['close', 'fgi_value'],
                  labels={'value': 'Value', 'variable': 'Metric'},
                  title="📈 BTC Price vs Fear & Greed Index",
                  template='plotly_dark')
    return fig

# Boxplot: Daily return distribution by FGI sentiment
def plot_return_boxplot(df):
    fig = px.box(df, x='fgi_sentiment_lag1', y='daily_return', color='fgi_sentiment_lag1',
                 title="📊 Daily Returns vs Sentiment (Lag-1)",
                 labels={'daily_return': 'Daily Return (%)', 'fgi_sentiment_lag1': 'Yesterday\'s Sentiment'},
                 template='plotly')
    return fig

# Correlation heatmap (matplotlib + seaborn)
def plot_corr_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    plt.title("📉 Correlation Matrix")
    return fig
