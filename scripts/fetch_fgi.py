import requests
import pandas as pd
from datetime import datetime

def fetch_fgi_data(limit=1000):
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"

    response = requests.get(url)
    data = response.json()

    fgi_list = data['data']

    df = pd.DataFrame(fgi_list)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['value'] = df['value'].astype(int)
    df = df.rename(columns={
        'value': 'fgi_value',
        'value_classification': 'fgi_sentiment',
        'timestamp': 'date'
    })

    df = df[['date', 'fgi_value', 'fgi_sentiment']].sort_values('date')
    return df

# Run if executed directly
if __name__ == "__main__":
    fgi_df = fetch_fgi_data()
    print(fgi_df.head())
    fgi_df.to_csv("data/fgi_data.csv", index=False)
    print("✅ Fear & Greed Index data saved to data/fgi_data.csv")
