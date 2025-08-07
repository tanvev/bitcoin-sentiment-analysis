# models/model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_logistic_model(df):
    # Select features and target
    features = ['fgi_value', 'fgi_value_lag1', 'volatility', 'sentiment_encoded']
    df_model = df.dropna(subset=features + ['target'])

    X = df_model[features]
    y = df_model['target']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Compute accuracy on training data
    accuracy = model.score(X_train, y_train)

    # Extract feature importance
    coefs = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    return model, scaler, accuracy, coefs
