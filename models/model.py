import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_logistic_model(df):
    df = df.copy()

    # Target: whether next day's return is positive
    df['target'] = (df['daily_return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    # Encode sentiment
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['fgi_sentiment_lag1'])

    X = df[['sentiment_encoded', 'daily_return', 'volatility_7d']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Show weights
    coefs = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0].round(4)
    })

    return acc, coefs
