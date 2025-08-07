# models/prediction.py

import os
import pandas as pd
from datetime import datetime

def load_or_create_prediction(run_model_func, df):
    model, scaler, accuracy, coefs = run_model_func(df)

    # Predict next-day direction using today’s row
    today_row = df.iloc[-1:][['fgi_value', 'fgi_value_lag1', 'volatility', 'sentiment_encoded']]
    if today_row.isnull().any().any():
        return None, 0.0, None

    X_today = scaler.transform(today_row)
    prediction = model.predict(X_today)[0]

    # Save historical prediction
    save_path = "data/predictions.csv"
    correct = "N/A"
    if len(df) >= 2:
        yesterday_close = df.iloc[-2]['close']
        today_close = df.iloc[-1]['close']
        true_movement = int(today_close > yesterday_close)
        correct = int(true_movement == prediction)

    new_row = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predicted_direction": prediction,
        "is_correct": correct,
        "accuracy": accuracy
    }])

    if os.path.exists(save_path):
        history = pd.read_csv(save_path)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    history.to_csv(save_path, index=False)

    return prediction, accuracy, coefs
