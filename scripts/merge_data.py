import pandas as pd

def merge_fgi_and_btc(fgi_path="data/fgi_data.csv", btc_path="data/btc_data.csv"):
    fgi = pd.read_csv(fgi_path, parse_dates=["date"])
    btc = pd.read_csv(btc_path, parse_dates=["date"])

    # Align by date (inner join keeps only overlapping days)
    merged = pd.merge(btc, fgi, on="date", how="inner")

    # Sort and reset index
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged

# Run if executed directly
if __name__ == "__main__":
    df = merge_fgi_and_btc()
    print(df.head())
    df.to_csv("data/merged_data.csv", index=False)
    print("✅ Merged BTC + FGI data saved to data/merged_data.csv")
