import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_logistic_model(df):
    df = df.copy()
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["fgi_sentiment_lag1"])

    df.dropna(subset=["daily_return", "volatility_7d"], inplace=True)

    X = df[["sentiment_encoded", "daily_return", "volatility_7d"]]
    y = (df["close"].shift(-1) > df["close"]).astype(int)[:-1]  # 1 if price goes up next day

    X = X.iloc[:-1]  # match shape with y
    model = LogisticRegression()
    model.fit(X, y)

    accuracy = model.score(X, y)

    # Predict tomorrow's direction
    latest = df.iloc[-1:][["sentiment_encoded", "daily_return", "volatility_7d"]]
    pred = model.predict(latest)[0]

    coefs = pd.DataFrame({
        "Feature": ["Sentiment", "Daily Return", "Volatility"],
        "Coefficient": model.coef_[0]
    })

    return accuracy, pred, coefs
