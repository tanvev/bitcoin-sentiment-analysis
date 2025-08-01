import pandas as pd


def generate_sentiment_summary(df):
    summary = df.groupby('fgi_sentiment_lag1').agg(
        avg_return=('daily_return', 'mean'),
        volatility=('daily_return', 'std'),
        win_rate=('daily_return', lambda x: (x > 0).mean())
    ).reset_index()

    summary.columns = ['Sentiment (Yesterday)', 'Avg Return (%)', 'Volatility', 'Win Rate']
    summary['Avg Return (%)'] = (summary['Avg Return (%)'] * 100).round(2)
    summary['Volatility'] = (summary['Volatility'] * 100).round(2)
    summary['Win Rate'] = (summary['Win Rate'] * 100).round(1).astype(str) + '%'

    return summary


def generate_observations(df):
    obs = []

    fear_df = df[df['fgi_sentiment_lag1'] == 'Extreme Fear']
    greed_df = df[df['fgi_sentiment_lag1'] == 'Extreme Greed']

    if not fear_df.empty and not greed_df.empty:
        avg_fear_ret = fear_df['daily_return'].mean() * 100
        avg_greed_ret = greed_df['daily_return'].mean() * 100

        obs.append(f"📉 **Extreme Fear** days saw an average return of **{avg_fear_ret:.2f}%**.")
        obs.append(f"📈 **Extreme Greed** days saw an average return of **{avg_greed_ret:.2f}%**.")

        if avg_greed_ret > avg_fear_ret:
            obs.append("💡 Market performs better during **greed** phases.")
        else:
            obs.append("💡 Market tends to crash during **greedy** phases — maybe due to overbuying.")

    return obs
