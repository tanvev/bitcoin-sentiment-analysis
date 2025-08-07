import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go

# Line plot: BTC price + FGI over time

def plot_price_vs_sentiment(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        name="BTC Closing Price",
        yaxis="y1",
        line=dict(color="royalblue")
    ))

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['fgi_value'],
        name="Fear & Greed Index",
        yaxis="y2",
        line=dict(color="orange")
    ))

    fig.update_layout(
        title="BTC Price vs Fear & Greed Index Over Time",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price (USD)", side="left"),
        yaxis2=dict(title="FGI Value", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h")
    )

    return fig



def plot_return_boxplot(df):
    fig = px.box(
        df,
        x="fgi_sentiment",
        y="daily_return",
        color="fgi_sentiment",
        title="Distribution of Daily Returns by Sentiment Category",
        labels={"fgi_sentiment": "Sentiment", "daily_return": "Daily Return"},
    )
    fig.update_layout(showlegend=False)
    return fig


# Correlation heatmap (matplotlib + seaborn)
def plot_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    return fig


def plot_price_with_moving_averages(df):
    df = df.copy()
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='BTC Close Price', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA_7'], mode='lines', name='7-Day MA', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['MA_30'], mode='lines', name='30-Day MA', line=dict(color='orange')))

    fig.update_layout(
        title="Bitcoin Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500
    )
    return fig

def plot_return_histogram(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['daily_return'], nbinsx=50, name='Daily Returns'))
    fig.update_layout(
        title='Distribution of Daily Returns',
        xaxis_title='Return',
        yaxis_title='Frequency'
    )
    return fig


def plot_feature_importance(coefs):
    import pandas as pd

    # Convert to DataFrame if not already
    if not isinstance(coefs, pd.DataFrame):
        coefs = pd.DataFrame(coefs, columns=["Feature", "Coefficient"])

    fig = px.bar(coefs, x="Feature", y="Coefficient", title="Logistic Regression Coefficients")
    fig.update_layout(xaxis_title="Feature", yaxis_title="Coefficient")
    return fig

def plot_volatility_trendline(df):
    fig = px.line(df, x="date", y="volatility_7d", title="7-Day Rolling Volatility")
    fig.update_traces(mode="lines+markers")
    return fig
