# Bitcoin Sentiment Analysis Dashboard

An interactive Streamlit dashboard that analyzes the impact of the Fear & Greed Index on Bitcoin (BTC) price movements, and predicts the next-day price direction using Logistic Regression.

---

## Overview

This project explores the relationship between market sentiment and Bitcoin trading behavior. Inspired by academic research, it includes:

- Visualizations of sentiment vs. price behavior  
- Logistic Regression model for next-day BTC price direction prediction  
- Feature importance insights via coefficient plots  
- Downloadable datasets and prediction tracker  
- Real-time BTC price updates with volatility trendlines  

---

## Live Demo

https://bitcoin-sentiment-analysis.streamlit.app/

---

## Features

| Feature                          | Description |
|----------------------------------|-------------|
| Sentiment vs Price Trends        | Correlates Bitcoin prices with Fear/Greed index |
| Model Prediction                 | Predicts next-day BTC price direction using logistic regression |
| Volatility Tracker               | 7-day rolling volatility chart |
| Feature Importance Chart         | Coefficients of the trained model |
| Historical Accuracy Tracker      | Tracks model predictions and accuracy over time |
| Live BTC Price Feed              | Fetched using yfinance |
| Dataset Download                 | Download filtered CSVs for your own analysis |

---

## Technologies Used

- Python 3.11
- Streamlit
- pandas, scikit-learn, plotly
- yfinance
- GitHub + Streamlit Cloud

---

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/bitcoin-sentiment-analysis.git
   cd bitcoin-sentiment-analysis
