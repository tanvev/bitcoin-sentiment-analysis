import pandas as pd

def create_features(merged_path="data/merged_data.csv"):
    df = pd.read_csv(merged_path, parse_dates=["date"])
    df = df.sort_values("date")

    # Daily returns (percentage)
    df['daily_return'] = df['close'].pct_change() * 100

    # Rolling 7-day volatility (standard deviation of returns)
    df['volatility_7d'] = df['daily_return'].rolling(window=7).std()

    # Lag features (yesterday’s FGI value & sentiment)
    df['fgi_value_lag1'] = df['fgi_value'].shift(1)
    df['fgi_sentiment_lag1'] = df['fgi_sentiment'].shift(1)

    # Drop rows with NaN due to lag/rolling
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# Run if executed directly
if __name__ == "__main__":
    features_df = create_features()
    print(features_df[["date", "close", "daily_return", "volatility_7d", "fgi_value", "fgi_value_lag1"]].tail())
    features_df.to_csv("data/featured_data.csv", index=False)
    print("✅ Feature-engineered data saved to data/featured_data.csv")
