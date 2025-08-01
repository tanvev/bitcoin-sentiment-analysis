@st.cache_data
def load_data():
    df = pd.read_csv("data/featured_data.csv", parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Ensure required columns exist
    if 'daily_return' not in df.columns:
        df['daily_return'] = df['price'].pct_change()

    if 'volatility' not in df.columns:
        df['volatility'] = df['daily_return'].rolling(window=5).std()

    if 'fgi_sentiment_lag1' not in df.columns:
        df['fgi_sentiment_lag1'] = df['fgi_sentiment'].shift(1)

    # Drop rows with missing values in required model features
    df.dropna(subset=['daily_return', 'volatility', 'fgi_sentiment_lag1'], inplace=True)

    return df
