import yfinance as yf
import pandas as pd


def fetch_btc_data(start="2018-01-01", end=None):
    btc = yf.download("BTC-USD", start=start, end=end, interval="1d", progress=False)

    # Clean and format
    btc = btc.reset_index()
    btc = btc.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    btc = btc[['date', 'open', 'high', 'low', 'close', 'volume']]
    btc['date'] = pd.to_datetime(btc['date'])
    return btc


# Run if executed directly
if __name__ == "__main__":
    btc_df = fetch_btc_data()
    print(btc_df.tail())
    btc_df.to_csv("data/btc_data.csv", index=False)
    print("✅ BTC price data saved to data/btc_data.csv")
